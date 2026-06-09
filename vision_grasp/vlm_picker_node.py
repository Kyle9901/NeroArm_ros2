#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped, PoseStamped
from cv_bridge import CvBridge
import tf2_ros
import tf2_geometry_msgs  # 用于支持TF转换
import message_filters
import cv2
import base64
import requests
import json
import numpy as np
import os
import threading


class VlmPickerNode(Node):
    def __init__(self, target_object):
        super().__init__('vlm_picker_node')
        self.bridge = CvBridge()
        self.target_object = target_object

        # === 1. 核心配置区域 ===
        self.api_key = os.environ.get("VLM_API_KEY", "")
        if not self.api_key:
            self.get_logger().warn(
                "⚠️ 环境变量 VLM_API_KEY 未设置，VLM 请求将失败！请 export VLM_API_KEY=***")
        self.api_url = os.environ.get(
            "VLM_API_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions")
        self.model_name = os.environ.get("VLM_MODEL", "qwen3.7-plus")
        self.prompt_template = (
            "Task: Find the physical object '{target}' in this image.\n"
            "Context: You are a robot eye-in-hand camera. "
            "Ignore the black robotic grippers at the bottom. "
            "Ignore the small holes/indentations on the black foam surface. Focus strictly on the actual 3D physical object.\n"
            "The image resolution is {width}x{height} pixels.\n\n"
            "=== VISUAL COORDINATE GRID HELP ===\n"
            "To help you output highly accurate bounding box coordinates, a visual grid has been overlaid on the image:\n"
            "- BLUE lines are the X-axis (width from 0 to {width}).\n"
            "- RED lines are the Y-axis (height from 0 to {height}).\n"
            "CRITICAL INSTRUCTION: Do NOT just output the exact numbers written on the grid lines! You must INTERPOLATE between the lines. "
            "For example, if an object edge is exactly halfway between the 200 and 250 lines, you MUST output 225. "
            "Ensure the bounding box is EXTREMELY TIGHT, touching the very outer edges of the physical object.\n\n"
            "Step 1: Classify the object into one category:\n"
            '- "color_block": solid single color, simple shape like cube/block (e.g. blue block, red cube)\n'
            '- "textured": complex patterns, non-uniform color, or printed labels\n'
            '- "reflective": glass, metal, bottle, or shiny surface\n'
            '- "other": none of the above\n\n'
            "Step 2: If category is 'color_block', describe its dominant color and approximate location.\n"
            "If category is NOT 'color_block', provide a tight bounding box.\n\n"
            
            "=== CRITICAL JSON RULES ===\n"
            "1. Return ONLY raw, syntactically valid JSON. No markdown block formatting (```json).\n"
            "2. The 'bbox' object MUST contain EXACTLY 4 key-value pairs: 'xmin', 'ymin', 'xmax', 'ymax'.\n"
            "3. DO NOT omit any keys. DO NOT group multiple numbers under one single key.\n\n"
            
            "=== PROHIBITED NEGATIVE EXAMPLE (NEVER OUTPUT THIS) ===\n"
            "❌ {{\"category\": \"reflective\", \"bbox\": {{\"xmin\": 446, 187, 536, 418}}, \"found\": true}}\n"
            "Why it is WRONG: The JSON syntax is completely broken! It puts 4 numbers under one 'xmin' key without other keys. This causes a syntax crash.\n\n"
            
            "=== REQUIRED POSITIVE EXAMPLE (MUST FOLLOW THIS EXACT FORMAT) ===\n"
            "✅ {{\"category\": \"reflective\", \"bbox\": {{\"xmin\": 446, \"ymin\": 187, \"xmax\": 536, \"ymax\": 418}}, \"found\": true}}\n"
            "Why it is CORRECT: Every single value has its own explicit key. It is a 100% valid JSON object.\n\n"
            
            "=== EXPECTED OUTPUT FORMAT FOR COLOR_BLOCK ===\n"
            "{{\"category\": \"color_block\", \"color\": \"blue\", \"alternative_colors\": [], \"location_hint\": \"center\", \"found\": true}}\n\n"
            
            "Strictly follow the rules. Do not include any extra chat text, thinking process, or keys."
        )

        # === 2. 状态变量 ===
        self.latest_color_img = None
        self.latest_depth_raw = None       # 对齐后的深度图 (640×480，与彩色图同分辨率)
        self.color_intrinsics = None       # 彩色相机内参 {fx, fy, cx, cy, width, height}
        self._lock = threading.Lock()      # 保护共享状态的锁
        self._vlm_processing = False       # 避免 VLM 请求并发重叠
        self._vlm_done = False             # 成功发布抓取目标后停止检测
        self._should_exit = False          # 成功发布后通知 main 退出

        # === 3. TF2 监听器（用于坐标转换） ===
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # === 4. 订阅相机相关话题 ===
        # 使用 message_filters 同步彩色图与硬件对齐深度图
        color_sub = message_filters.Subscriber(self, Image, '/camera/color/image_raw')
        depth_sub = message_filters.Subscriber(self, Image, '/camera/depth/image_raw')
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [color_sub, depth_sub],
            queue_size=10,
            slop=0.05)
        self.sync.registerCallback(self.synced_callback)

        self.color_info_sub = self.create_subscription(
            CameraInfo, '/camera/color/camera_info', self.color_info_callback, 10)

        self.grasp_target_pub = self.create_publisher(PoseStamped, '/grasp/target', 10)
        self.process_timer = self.create_timer(5.0, self.timer_callback)
        self.get_logger().info(
            f"VLM 节点已启动，目标: {self.target_object}，模型: {self.model_name}")

    def synced_callback(self, color_msg, depth_msg):
        """同步回调：彩色图与硬件对齐深度图（同分辨率、同坐标系）。"""
        with self._lock:
            self.latest_color_img = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")
            self.latest_depth_raw = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1")

    def color_info_callback(self, msg):
        with self._lock:
            self.color_intrinsics = {
                'fx': msg.k[0], 'fy': msg.k[4],
                'cx': msg.k[2], 'cy': msg.k[5],
                'width': msg.width, 'height': msg.height
            }

    def timer_callback(self):
        with self._lock:
            if self._vlm_done or self._vlm_processing:
                return

            color_ok = self.latest_color_img is not None
            depth_ok = self.latest_depth_raw is not None
            intr_ok = self.color_intrinsics is not None

        if not (color_ok and depth_ok and intr_ok):
            missing = []
            if not color_ok: missing.append("彩色图")
            if not depth_ok: missing.append("深度图")
            if not intr_ok: missing.append("相机内参")
            self.get_logger().warn(f"等待数据: {', '.join(missing)}...")
            return

        # VLM 请求放到后台线程；失败后允许下一轮 timer 重试，成功发布后停止
        with self._lock:
            self._vlm_processing = True
        thread = threading.Thread(target=self._process_vlm_request, daemon=True)
        thread.start()

    def _process_vlm_request(self):
        """在后台线程中执行深度对齐 + VLM 请求 + 坐标转换，不阻塞 ROS 主线程。"""
        _response_text_for_debug = "N/A"
        try:
            # 获取同步的彩色图与硬件对齐深度图（同分辨率同坐标系）
            with self._lock:
                depth_raw = self.latest_depth_raw.copy()
                color_img = self.latest_color_img.copy()
                color_intr = self.color_intrinsics.copy()

            # === 为发给 VLM 的图像加上坐标网格，辅助精确定位 ===
            vlm_input_img = self._add_coordinate_grid(color_img, step=50)

            # 1. 将带网格的图像编码为 Base64
            ok, buffer = cv2.imencode('.jpg', vlm_input_img)
            if not ok:
                self.get_logger().error("图像编码失败，等待重试。")
                return
            base64_image = base64.b64encode(buffer).decode('utf-8')

            # 2. 构建大模型视觉请求 Payload
            img_h, img_w = color_img.shape[:2]
            prompt = self.prompt_template.format(
                target=self.target_object,
                width=img_w,
                height=img_h)
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            payload = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }}
                        ]
                    }
                ],
                "max_tokens": 300
            }

            # 3. 发送请求（已在后台线程中，不会阻塞 ROS 主循环）
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=60.0)
            _response_text_for_debug = response.text
            if response.status_code != 200:
                self.get_logger().error(
                    f"VLM API 错误 status={response.status_code}, "
                    f"响应: {_response_text_for_debug[:500]}")
                return
            res_json = response.json()
            ai_output = res_json['choices'][0]['message']['content'].strip()

            # 清理可能存在的 Markdown 标记 (如 ```json ... ```)
            if ai_output.startswith("```"):
                ai_output = ai_output.split("```")[1]
                if ai_output.startswith("json"):
                    ai_output = ai_output[4:]
                ai_output = ai_output.strip()

            # 解析 JSON
            try:
                box = json.loads(ai_output)
            except json.JSONDecodeError as e:
                self.get_logger().error(
                    f"VLM 输出不是有效 JSON: {e}\n原始内容: {ai_output}")
                return

            if not box.get("found", True):
                self.get_logger().info(f"未找到目标: {self.target_object}，等待重试。")
                return

            # ── 路由：根据 category 选择检测方式 ──
            category = box.get("category", "other")
            xmin = ymin = xmax = ymax = None
            detect_source = "VLM"

            if category == "color_block":
                color_name = box.get("color", "")
                location_hint = box.get("location_hint", "")
                alt_colors = box.get("alternative_colors", [])
                self.get_logger().info(
                    f"VLM 分类: color_block, 颜色={color_name}, "
                    f"备选={alt_colors}, 位置={location_hint}")

                # 尝试主颜色
                bbox_cv = self._detect_by_color(color_img, color_name, location_hint)
                # 主颜色失败时尝试备选颜色
                if bbox_cv is None and alt_colors:
                    for alt in alt_colors:
                        bbox_cv = self._detect_by_color(color_img, alt, location_hint)
                        if bbox_cv is not None:
                            break

                if bbox_cv is not None:
                    xmin, ymin, xmax, ymax = bbox_cv
                    detect_source = "CV"
                else:
                    self.get_logger().warn("OpenCV 颜色检测失败，fallback 到 VLM bbox。")

            # 如果上面没拿到 bbox（非 color_block 或 OpenCV 失败），用 VLM 给的 bbox
            if xmin is None:
                if "bbox" not in box:
                    self.get_logger().error("VLM 返回缺少 bbox 字段，等待重试。")
                    return
                b = box["bbox"]
                if not all(k in b for k in ("xmin", "ymin", "xmax", "ymax")):
                    self.get_logger().error("VLM bbox 字段不完整，等待重试。")
                    return
                xmin = int(b["xmin"])
                ymin = int(b["ymin"])
                xmax = int(b["xmax"])
                ymax = int(b["ymax"])

            depth_h, depth_w = depth_raw.shape[:2]
            xmin, xmax = sorted((xmin, xmax))
            ymin, ymax = sorted((ymin, ymax))
            xmin = max(0, min(xmin, depth_w - 1))
            xmax = max(0, min(xmax, depth_w - 1))
            ymin = max(0, min(ymin, depth_h - 1))
            ymax = max(0, min(ymax, depth_h - 1))

            if xmax <= xmin or ymax <= ymin:
                self.get_logger().error(
                    f"bbox 无效: ({xmin},{ymin})-({xmax},{ymax})，等待重试。")
                return

            u = int((xmin + xmax) / 2)
            v = int((ymin + ymax) / 2)
            self.get_logger().info(
                f"已定位 ({detect_source}) bbox=({xmin},{ymin})-({xmax},{ymax}), center=({u},{v})")

            # 5. 直接在对齐深度图上读取物块深度（硬件已对齐，同分辨率同坐标系）
            box_w = xmax - xmin + 1
            box_h = ymax - ymin + 1
            margin_x = max(1, int(box_w * 0.25))
            margin_y = max(1, int(box_h * 0.25))
            roi_xmin = min(xmin + margin_x, xmax)
            roi_xmax = max(xmax - margin_x, xmin)
            roi_ymin = min(ymin + margin_y, ymax)
            roi_ymax = max(ymax - margin_y, ymin)

            roi = depth_raw[roi_ymin:roi_ymax + 1, roi_xmin:roi_xmax + 1]
            valid_depths = roi[roi > 0]

            if len(valid_depths) == 0:
                self.get_logger().warn("bbox 中心无有效深度，改用中心 11×11。")
                depth_h, depth_w = depth_raw.shape[:2]
                roi = depth_raw[max(0, v-5):min(depth_h, v+6),
                                max(0, u-5):min(depth_w, u+6)]
                valid_depths = roi[roi > 0]

            if len(valid_depths) == 0:
                self.get_logger().error("目标深度无效，等待重试。")
                return

            depth_mm = float(np.median(valid_depths))
            self.get_logger().info(f"深度={depth_mm:.1f}mm，有效点={len(valid_depths)}")

            z_c = depth_mm / 1000.0
            fx = color_intr['fx']
            fy = color_intr['fy']
            cx = color_intr['cx']
            cy = color_intr['cy']
            x_c = (u - cx) * z_c / fx
            y_c = (v - cy) * z_c / fy

            self.get_logger().info(f"[CAM-3D] x_c={x_c:.4f}, y_c={y_c:.4f}, z_c={z_c:.4f}")
            # 6. 利用 TF2 将坐标从相机转换到机器人底座 base_link
            point_in_camera = PointStamped()
            point_in_camera.header.frame_id = "camera_color_optical_frame"
            # stamp 设为当前时间，但 lookup_transform 使用 Time()（最新可用变换）。
            point_in_camera.header.stamp = self.get_clock().now().to_msg()
            point_in_camera.point.x = x_c
            point_in_camera.point.y = y_c
            point_in_camera.point.z = z_c

            # 阻塞等待直到 TF 树中有可用的变换（在后台线程中，不阻塞 ROS spin）
            transform = self.tf_buffer.lookup_transform(
                "base_link",
                "camera_color_optical_frame",
                rclpy.time.Time(),  # 使用最新可用变换
                timeout=rclpy.duration.Duration(seconds=1.0)
            )

            # 执行空间坐标变换
            point_in_base = tf2_geometry_msgs.do_transform_point(
                point_in_camera, transform)

            self._save_debug_image(
                vlm_input_img,
                (xmin, ymin, xmax, ymax),
                (u, v),
                (roi_xmin, roi_ymin, roi_xmax, roi_ymax),
                depth_mm,
                point_in_base,
                source=detect_source)

            # === 🎯 发布抓取目标到 /grasp/target，联动 grasp_executor ===
            grasp_target = PoseStamped()
            grasp_target.header.frame_id = "base_link"
            grasp_target.header.stamp = self.get_clock().now().to_msg()
            grasp_target.pose.position.x = point_in_base.point.x
            grasp_target.pose.position.y = point_in_base.point.y
            grasp_target.pose.position.z = point_in_base.point.z
            self.grasp_target_pub.publish(grasp_target)
            with self._lock:
                self._vlm_done = True
                self._should_exit = True

            self.get_logger().info(
                f"已发布 /grasp/target: x={point_in_base.point.x:.4f}, "
                f"y={point_in_base.point.y:.4f}, z={point_in_base.point.z:.4f}")
            self.get_logger().info("识别完成，节点退出。")

        except requests.exceptions.Timeout:
            self.get_logger().error("VLM 请求超时，等待重试。")
        except requests.exceptions.RequestException as e:
            self.get_logger().error(f"VLM 网络异常，等待重试: {e}")
        except KeyError as e:
            self.get_logger().error(
                f"VLM 返回结构异常(KeyError={e})。\n"
                f"原始响应: {_response_text_for_debug[:800]}")
        except Exception as e:
            self.get_logger().error(
                f"处理失败: {type(e).__name__}: {e}\n"
                f"原始响应: {_response_text_for_debug[:800]}")
        finally:
            with self._lock:
                self._vlm_processing = False

    def _add_coordinate_grid(self, img, step=100):
        """
        在图像上覆盖带有像素坐标数字的网格，辅助 VLM 进行空间定位。
        横轴（X）用蓝色，纵轴（Y）用红色。
        """
        annotated_img = img.copy()
        h, w = annotated_img.shape[:2]

        # 画垂直线 (X轴坐标)，蓝色
        for x in range(0, w, step):
            cv2.line(annotated_img, (x, 0), (x, h), (255, 0, 0), 1)
            cv2.putText(annotated_img, str(x), (x + 5, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            cv2.putText(annotated_img, str(x), (x + 5, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # 画水平线 (Y轴坐标)，红色
        for y in range(0, h, step):
            cv2.line(annotated_img, (0, y), (w, y), (0, 0, 255), 1)
            cv2.putText(annotated_img, str(y), (5, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(annotated_img, str(y), (w - 45, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # 在每个交点标注完整坐标 (x,y)
        for x in range(0, w, step):
            for y in range(0, h, step):
                text = f"({x},{y})"
                # 黑色描边增强可读性
                cv2.putText(annotated_img, text, (x + 2, y + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 0, 0), 2)
                cv2.putText(annotated_img, text, (x + 2, y + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255), 1)

        return annotated_img

    def _save_debug_image(self, color_img, bbox, center, roi_box, depth_mm, point_in_base, source=""):
        debug_img = color_img.copy()
        xmin, ymin, xmax, ymax = bbox
        u, v = center
        roi_xmin, roi_ymin, roi_xmax, roi_ymax = roi_box

        # 根据来源选择框颜色：CV=绿色, VLM=青色
        box_color = (0, 255, 0) if source == "CV" else (255, 255, 0)
        cv2.rectangle(debug_img, (xmin, ymin), (xmax, ymax), box_color, 2)
        cv2.rectangle(debug_img, (roi_xmin, roi_ymin), (roi_xmax, roi_ymax), (255, 0, 0), 2)
        cv2.circle(debug_img, (u, v), 5, (0, 0, 255), -1)
        if source:
            cv2.putText(debug_img, source, (xmax + 5, ymin + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

        lines = [
            f"target: {self.target_object}",
            f"source: {source}",
            f"bbox: ({xmin},{ymin})-({xmax},{ymax}) center=({u},{v})",
            f"depth: {depth_mm:.0f} mm",
            f"base: x={point_in_base.point.x:.4f}, y={point_in_base.point.y:.4f}, z={point_in_base.point.z:.4f}",
        ]
        y0 = 30
        for i, text in enumerate(lines):
            y = y0 + i * 28
            cv2.putText(debug_img, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(debug_img, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 2, cv2.LINE_AA)

        import os as _os
        debug_dir = _os.environ.get("VLM_DEBUG_DIR", _os.path.join(_os.path.expanduser("~"), "vlm_grasp_debug"))
        _os.makedirs(debug_dir, exist_ok=True)
        debug_path = _os.path.join(debug_dir, "vlm_picker_debug.jpg")
        if not cv2.imwrite(debug_path, debug_img):
            self.get_logger().warn(f"debug图保存失败: {debug_path}")

    # ─────────────────────────── OpenCV 颜色检测 ───────────────────────────
    _COLOR_HSV_RANGES = {
        'blue':   [((90, 80, 40),  (140, 255, 255))],
        'red':    [((0, 100, 50),   (10, 255, 255)),
                   ((170, 100, 50), (180, 255, 255))],
        'green':  [((35, 80, 40),   (85, 255, 255))],
        'yellow': [((20, 100, 50),  (35, 255, 255))],
        'purple': [((120, 80, 40),  (160, 255, 255))],
        'orange': [((10, 100, 50),  (20, 255, 255))],
        'cyan':   [((80, 80, 40),   (100, 255, 255))],
    }

    def _detect_by_color(self, color_img, color_name, location_hint):
        """用 OpenCV HSV 颜色检测找目标 bbox。返回 (xmin,ymin,xmax,ymax) 或 None。"""
        hsv = cv2.cvtColor(color_img, cv2.COLOR_BGR2HSV)
        h, w = color_img.shape[:2]
        img_area = h * w

        # 1. 准备 HSV 范围：主颜色 + alternative 颜色
        ranges = []
        color_lower = color_name.lower()
        if color_lower in self._COLOR_HSV_RANGES:
            ranges.extend(self._COLOR_HSV_RANGES[color_lower])
        # 尝试相近颜色（如 blue 失败后试 purple/cyan）
        for alt in ['purple', 'cyan'] if color_lower == 'blue' else []:
            if alt in self._COLOR_HSV_RANGES:
                ranges.extend(self._COLOR_HSV_RANGES[alt])

        if not ranges:
            self.get_logger().warn(f"未知颜色 '{color_name}'，无 HSV 映射。")
            return None

        # 2. 合并所有颜色范围的 mask
        combined_mask = None
        for (lower, upper) in ranges:
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            if combined_mask is None:
                combined_mask = mask
            else:
                combined_mask = cv2.bitwise_or(combined_mask, mask)

        # 3. 形态学去噪
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # 4. 找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 200:                       # 下限：噪点
                continue
            if area > img_area * 0.3:            # 上限：不可能是单物体
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            cx, cy = x + bw / 2, y + bh / 2
            aspect = bw / max(bh, 1)
            if aspect < 0.2 or aspect > 5.0:     # 太扁或太瘦
                continue
            candidates.append({
                'area': area, 'x': x, 'y': y,
                'xmax': x + bw, 'ymax': y + bh,
                'cx': cx, 'cy': cy,
            })

        if not candidates:
            self.get_logger().warn(f"OpenCV 未找到颜色 '{color_name}' 的候选区域。")
            return None

        # 5. 位置过滤（根据 VLM 的 location_hint）
        filtered = self._filter_by_location(candidates, location_hint, w, h)
        if not filtered:
            self.get_logger().warn("位置过滤后无候选，尝试所有候选。")
            filtered = candidates

        # 6. 选面积最大的
        best = max(filtered, key=lambda c: c['area'])
        self.get_logger().info(
            f"OpenCV 颜色检测: 颜色={color_name}, "
            f"bbox=({best['x']},{best['y']})-({best['xmax']},{best['ymax']}), "
            f"面积={best['area']:.0f}")
        return (best['x'], best['y'], best['xmax'], best['ymax'])

    @staticmethod
    def _filter_by_location(candidates, hint, img_w, img_h):
        if not hint or hint == 'unknown':
            return candidates
        hint = hint.lower().replace('-', ' ').strip()
        result = []
        for c in candidates:
            cx, cy = c['cx'], c['cy']
            ok = True
            if 'left' in hint and cx > img_w * 0.6:
                ok = False
            if 'right' in hint and cx < img_w * 0.4:
                ok = False
            if 'top' in hint and cy > img_h * 0.6:
                ok = False
            if 'bottom' in hint and cy < img_h * 0.4:
                ok = False
            if 'center' in hint:
                if not (img_w * 0.25 < cx < img_w * 0.75 and img_h * 0.25 < cy < img_h * 0.75):
                    ok = False
            if ok:
                result.append(c)
        return result

    def should_exit(self):
        with self._lock:
            return self._should_exit


def main(args=None):
    rclpy.init(args=args)

    target_object = input("请输入要识别的物体，例如 blue block: ").strip()
    if not target_object:
        target_object = "blue block"
        print(f"未输入目标物体，默认识别: {target_object}")
    else:
        print(f"将识别目标物体: {target_object}")

    node = VlmPickerNode(target_object)
    try:
        while rclpy.ok() and not node.should_exit():
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()