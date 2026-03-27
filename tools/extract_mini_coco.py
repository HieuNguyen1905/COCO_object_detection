import os
import json
import random
import shutil
from tqdm import tqdm

def create_mini_coco(
    orig_json_path, 
    orig_images_dir, 
    out_json_path, 
    out_images_dir, 
    num_images=100
):
    print(f"[INFO] Đang đọc file COCO gốc: {orig_json_path} ...")
    with open(orig_json_path, 'r') as f:
        coco_data = json.load(f)

    print(f"[INFO] Tổng số ảnh gốc: {len(coco_data['images'])}")
    print(f"[INFO] Đang chọn ngẫu nhiên {num_images} ảnh...")
    
    # 1. Chọn ngẫu nhiên N bức ảnh
    selected_images = random.sample(coco_data['images'], min(num_images, len(coco_data['images'])))
    selected_image_ids = set([img['id'] for img in selected_images])

    # 2. Lọc ra các annotations (nhãn) chỉ thuộc về các ảnh đã chọn
    print("[INFO] Đang lọc annotations tương ứng...")
    selected_annotations = [
        ann for ann in coco_data['annotations'] 
        if ann['image_id'] in selected_image_ids
    ]

    # 3. Tạo data structure mới cho Mini COCO
    mini_coco_data = {
        "info": coco_data.get("info", {}),
        "licenses": coco_data.get("licenses", []),
        "images": selected_images,
        "annotations": selected_annotations,
        "categories": coco_data.get("categories", []) # Giữ nguyên toàn bộ danh mục classes
    }

    # 4. Lưu file JSON mới
    os.makedirs(os.path.dirname(out_json_path), exist_ok=True)
    with open(out_json_path, 'w') as f:
        json.dump(mini_coco_data, f)
    print(f"[INFO] Đã lưu file Mini JSON tại: {out_json_path}")
    print(f"[INFO] Số lượng annotations trong Mini Dataset: {len(selected_annotations)}")

    # 5. Copy ảnh vật lý sang thư mục mới
    print("[INFO] Đang copy ảnh sang thư mục mini...")
    os.makedirs(out_images_dir, exist_ok=True)
    
    for img_info in tqdm(selected_images):
        src_img_path = os.path.join(orig_images_dir, img_info['file_name'])
        dst_img_path = os.path.join(out_images_dir, img_info['file_name'])
        
        # Chỉ copy nếu file gốc thực sự tồn tại
        if os.path.exists(src_img_path):
            shutil.copy2(src_img_path, dst_img_path)
        else:
            print(f"[WARNING] Không tìm thấy ảnh: {src_img_path}")

    print("[INFO] HOÀN TẤT TRÍCH XUẤT MINI COCO!")

if __name__ == "__main__":
    # Tự động tìm thư mục root của project
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)  # Lên 1 cấp từ tools/

    # Train set
    ORIG_TRAIN_JSON = os.path.join(project_root, "data/annotations/instances_train2017.json")
    ORIG_TRAIN_IMAGES = os.path.join(project_root, "data/raw/train2017")
    MINI_TRAIN_JSON = os.path.join(project_root, "data/annotations/mini_instances_train2017.json")
    MINI_TRAIN_IMAGES = os.path.join(project_root, "data/raw/mini_train2017")

    # Validation set
    ORIG_VAL_JSON = os.path.join(project_root, "data/annotations/instances_val2017.json")
    ORIG_VAL_IMAGES = os.path.join(project_root, "data/raw/val2017")
    MINI_VAL_JSON = os.path.join(project_root, "data/annotations/mini_instances_val2017.json")
    MINI_VAL_IMAGES = os.path.join(project_root, "data/raw/mini_val2017")

    print("\n" + "="*60)
    print("TRÍCH XUẤT MINI TRAIN SET")
    print("="*60)
    create_mini_coco(
        orig_json_path=ORIG_TRAIN_JSON,
        orig_images_dir=ORIG_TRAIN_IMAGES,
        out_json_path=MINI_TRAIN_JSON,
        out_images_dir=MINI_TRAIN_IMAGES,
        num_images=100  # 100 ảnh cho training
    )

    print("\n" + "="*60)
    print("TRÍCH XUẤT MINI VALIDATION SET")
    print("="*60)
    create_mini_coco(
        orig_json_path=ORIG_VAL_JSON,
        orig_images_dir=ORIG_VAL_IMAGES,
        out_json_path=MINI_VAL_JSON,
        out_images_dir=MINI_VAL_IMAGES,
        num_images=50  # 50 ảnh cho validation (ít hơn train)
    )

    print("\n" + "="*60)
    print("HOÀN TẤT TẤT CẢ!")
    print("="*60)
    print(f"Train set: {MINI_TRAIN_JSON}")
    print(f"Val set: {MINI_VAL_JSON}")