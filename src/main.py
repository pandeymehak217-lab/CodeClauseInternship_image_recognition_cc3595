import argparse
from utils import read_image, show_image, save_image
from detector import detect_faces, detect_objects, draw_boxes

def main():
    parser = argparse.ArgumentParser(description="Image Recognition using OpenCV")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument(
        "--mode",
        choices=["face", "object"],
        default="face",
        help="Detection mode"
    )
    parser.add_argument(
        "--save",
        help="Path to save output image"
    )

    args = parser.parse_args()

    # Read image
    image = read_image(args.image)

    # Detection
    if args.mode == "face":
        boxes = detect_faces(image)
        output = draw_boxes(image, boxes, "Face")
    else:
        boxes = detect_objects(image)
        output = draw_boxes(image, boxes, "Object")

    print(f"Detected {len(boxes)} {args.mode}(s)")

    # Show or save result
    if args.save:
        save_image(args.save, output)
        print(f"Result saved to {args.save}")
    else:
        show_image("Detection Result", output)

if __name__ == "__main__":
    main()
