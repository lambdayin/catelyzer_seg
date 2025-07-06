import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm
import glob

import torch
from utils.data_loading import SelfDataset
from unet import UNet

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, help="Checkpoint .pth file for unet.")
    parser.add_argument('--input-dir', default='./test_0527/imgdata', type=str, help="Input directory containing images.")
    parser.add_argument('--output-dir', default='./test_0527/res', type=str, help="Output directory for results.")
    parser.add_argument('--grid-len', default=36, type=float, help="The physical length of each grid(mm).")
    parser.add_argument('--grid-pixel', default=651, type=float, help="Pixel distance of each grid.")
    parser.add_argument('--image-exts', default=['jpg', 'jpeg', 'png', 'bmp', 'tiff'], nargs='+', 
                       help="Image file extensions to process.")
    return parser.parse_args()

def get_image_files(input_dir, extensions):
    image_files = []
    for ext in extensions:
        pattern = os.path.join(input_dir, f"*.{ext}")
        image_files.extend(glob.glob(pattern, recursive=False))
        pattern = os.path.join(input_dir, f"*.{ext.upper()}")
        image_files.extend(glob.glob(pattern, recursive=False))
    return sorted(list(set(image_files)))

def pre_process_single_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")
    
    yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    mask_filter = yuv[:, :, 0] < 15
    mask_filter = mask_filter.astype(np.uint8)
    mask_eroded = cv2.dilate(mask_filter, np.ones((5, 5), np.uint8), iterations=2)
    mask_eroded = cv2.erode(mask_eroded, np.ones((5, 5), np.uint8), iterations=2)
        
    image[mask_eroded == 1] = [255, 255, 255]
    return image, mask_eroded

def inference_unet_batch(net, device, image_path):
    scale_factor = 0.5
    img_ori = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img_ori is None:
        raise ValueError(f"Cannot read image: {image_path}")
    net.eval()
    img = torch.from_numpy(SelfDataset.preprocess(None, img_ori, scale_factor, is_mask=False))    
    img = img.unsqueeze(0)           
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()        
        mask = output.argmax(dim=1)
        output = (mask[0] * 255).squeeze().numpy().astype(np.uint8)
    return output

def post_process_single_image(image_path, mm_per_pixel, mask_eroded, mask_unet):
    orig_img = cv2.imread(image_path)
    if orig_img is None:
        raise ValueError(f"Cannot read image: {image_path}")
    mask_eroded  = 1 - mask_eroded
    mask_unet = cv2.resize(mask_unet, (orig_img.shape[1], orig_img.shape[0]), interpolation=cv2.INTER_NEAREST)
    mask_check = mask_unet & mask_eroded
    mask_check = mask_check.astype(np.uint8)
    # mask_check = cv2.erode(mask_check, np.ones((3, 3), np.uint8), iterations=2)
    # mask_check = cv2.dilate(mask_check, np.ones((3, 3), np.uint8), iterations=2)
    # mask_check = cv2.dilate(mask_check, np.ones((3, 3), np.uint8), iterations=1)
    # mask_check = cv2.erode(mask_check, np.ones((3, 3), np.uint8), iterations=1)
    _, labeled_mask = cv2.connectedComponents(mask_check)
    labeled_mask = np.uint8(labeled_mask)

    contours, __ = cv2.findContours(labeled_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        min_rect = cv2.minAreaRect(contour)
        catelyzer_length = max(min_rect[1])
        catelyzer_length_mm = round(catelyzer_length * mm_per_pixel)
        rect_points = cv2.boxPoints(min_rect)
        rect_points = np.intp(rect_points)
        cv2.drawContours(orig_img, [contour], -1, (0, 0, 255), 2)
        # cv2.drawContours(orig_img, [rect_points], -1, (0, 0, 255), 2)
        # cv2.putText(orig_img, str(catelyzer_length_mm), (int(min_rect[0][0]),  int(min_rect[0][1])), 0, 0.5, (0, 255, 0), 2, 16)
    return orig_img


def process_single_image(image_path, net, device, mm_per_pixel, output_dir):
    try:
        _, mask_eroded = pre_process_single_image(image_path)
        mask_unet = inference_unet_batch(net, device, image_path)
        vis_image = post_process_single_image(image_path, mm_per_pixel, mask_eroded, mask_unet)
        filename = os.path.basename(image_path)
        name, ext = os.path.splitext(filename)
        output_path = os.path.join(output_dir, f"{name}_result{ext}")
        cv2.imwrite(output_path, vis_image)
        
        return True, output_path
    except Exception as e:
        return False, str(e)

def main():
    args = parse_args()
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist!")
        return
    os.makedirs(args.output_dir, exist_ok=True)
    image_files = get_image_files(args.input_dir, args.image_exts)
    if not image_files:
        print(f"No image files found in '{args.input_dir}' with extensions: {args.image_exts}")
        return
    print(f"Found {len(image_files)} images to process")

    len_per_pixel = float(args.grid_len/ args.grid_pixel)

    print("Loading UNet model...")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    net = UNet(n_channels=3, n_classes=2, bilinear=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device=device)
    
    try:
        state_dict = torch.load(args.model, map_location=device)
        _ = state_dict.pop('mask_values', [0, 1])
        net.load_state_dict(state_dict)
        print(f'Model loaded successfully! Using device: {device}')
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    successful = 0
    failed = 0
    failed_files = []

    print("Processing images...")
    for image_path in tqdm(image_files, desc="Processing", unit="image"):
        success, result = process_single_image(image_path, net, device, len_per_pixel, args.output_dir)
        
        if success:
            successful += 1
        else:
            failed += 1
            failed_files.append((image_path, result))
            print(f"\nFailed to process {image_path}: {result}")
    
    print(f"\n{'='*50}")
    print(f"Processing completed!")
    print(f"Total images: {len(image_files)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Results saved to: {args.output_dir}")
    
    if failed_files:
        print(f"\nFailed files:")
        for file_path, error in failed_files:
            print(f"  - {file_path}: {error}")

if __name__ == '__main__':
    main()
    
