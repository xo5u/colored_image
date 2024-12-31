import subprocess
import numpy as np
import cv2

import sec


def load_images(image_path1, image_path2):
    """
    Load two images from the specified file paths.
    """
    try:
        img1 = cv2.imread(image_path1)
        img2 = cv2.imread(image_path2)
        if img1 is None or img2 is None:
            raise ValueError("One or both images could not be loaded. Check file paths.")
        print("Images loaded successfully.")
        return img1, img2
    except Exception as e:
        print(f"Error loading images: {e}")
        return None, None

def show_images_side_by_side(img1, img2):
    """
    Display two images side by side in separate windows.
    """
    if img1 is None or img2 is None:
        print("Images are not loaded. Please load them first.")
        return

    cv2.imshow("Image 1", img1)
    cv2.imshow("Image 2", img2)
    cv2.moveWindow("Image 1", 0, 0)  # Position first window at the top-left corner
    cv2.moveWindow("Image 2", img1.shape[1], 0)  # Position second window to the right
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def cross_dissolve_images(img1, img2, steps=30):
    """
    Perform a cross-dissolve transition between two images.
    """
    if img1 is None or img2 is None:
        print("Images are not loaded. Please load them first.")
        return

    for step in range(steps + 1):
        alpha = step / steps
        blended_image = cv2.addWeighted(img1, 1 - alpha, img2, alpha, 0)
        cv2.imshow('Cross Dissolve Transition', blended_image)
        cv2.waitKey(100)  # Display each transition step briefly

    cv2.waitKey(500)  # Display the final blended image briefly
    cv2.destroyAllWindows()


def apply_pseudo_coloring(img):
    """
    Apply pseudo-coloring to a grayscale version of the image.
    """
    if img is None:
        print("Image is not loaded. Please load an image first.")
        return

    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    color_bins = [0, 51, 102, 153, 204, 255]
    colors = [
        [0, 0, 255], [0, 255, 0], [255, 0, 0],
        [255, 255, 0], [255, 0, 255], [0, 255, 255]
    ]

    pseudo_colored_image = np.zeros((gray_image.shape[0], gray_image.shape[1], 3), dtype=np.uint8)

    # Iterate through color_bins, but stop before the last one
    for i in range(len(color_bins) - 1):
        mask = (gray_image >= color_bins[i]) & (gray_image < color_bins[i + 1])
        pseudo_colored_image[mask] = colors[i]

    cv2.imshow("Pseudo-colored Image", pseudo_colored_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def color_segmentation(img):
    """
    Allow the user to click on the image to segment colors based on selected pixel.
    """
    if img is None:
        print("Image is not loaded. Please load an image first.")
        return

    def click_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            selected_color = img[y, x]
            hsv_color = cv2.cvtColor(np.uint8([[selected_color]]), cv2.COLOR_BGR2HSV)
            hue = hsv_color[0][0][0]

            lower_bound = np.array([max(0, hue - 10), 50, 50])
            upper_bound = np.array([min(179, hue + 10), 255, 255])

            hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv_img, lower_bound, upper_bound)
            segmented_img = cv2.bitwise_and(img, img, mask=mask)

            cv2.imshow("Segmented Image", segmented_img)

    cv2.imshow("Click to Segment Color", img)
    cv2.setMouseCallback("Click to Segment Color", click_callback)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def resize_images_to_match(img1, img2):
    """
    Resize two images to the smallest common dimensions.
    """
    if img1 is None or img2 is None:
        print("Images are not loaded. Please load them first.")
        return None, None

    min_height, min_width = min(img1.shape[0], img2.shape[0]), min(img1.shape[1], img2.shape[1])
    img1_resized = cv2.resize(img1, (min_width, min_height))
    img2_resized = cv2.resize(img2, (min_width, min_height))
    print("Images resized to the same dimensions.")
    return img1_resized, img2_resized

def display_menu():
    print("\n===== Image Processing Menu =====")
    print("1. Load Images")
    print("2. Show Images Side-by-Side")
    print("3. Resize Images")
    print("4. Cross Dissolve (Blend Images)")
    print("5. Apply Pseudo Coloring")
    print("6. Perform Color Segmentation")
    print("7. Detections")
    print("8. Exit")

def main():
    img1, img2 = None, None
    image_path2 = 'image1.png'
    image_path1 = 'image2.png'

    while True:
        display_menu()
        choice = input("Enter your choice: ")

        if choice == '1':
            img1, img2 = load_images(image_path1, image_path2)

        elif choice == '2':
            show_images_side_by_side(img1, img2)

        elif choice == '3':
            img1, img2 = resize_images_to_match(img1, img2)

        elif choice == '4':
            cross_dissolve_images(img1, img2)

        elif choice == '5':
            apply_pseudo_coloring(img1)

        elif choice == '6':
            color_segmentation(img2)


        elif choice == '7':
            print("Exiting program.")
            break
        else:
            print("Invalid choice. Please enter a number from 1 to 8.")

if __name__ == "__main__":
    main()
