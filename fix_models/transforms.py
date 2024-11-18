import torchvision.transforms as T
from torchvision.transforms import v2

class RecenterTransform:
    def __init__(self, center_x, center_y, output_size):
        self.center_x = center_x
        self.center_y = center_y
        self.output_size = output_size

    def __call__(self, image):
        width, height = image.size
        left = max(0, self.center_x - self.output_size[0] // 2)
        top = max(0, self.center_y - self.output_size[1] // 2)
        right = min(width, left + self.output_size[0])
        bottom = min(height, top + self.output_size[1])

        cropped_image = image.crop((left, top, right, bottom))

        if cropped_image.size != self.output_size:
            cropped_image = T.functional.pad(cropped_image, padding=(
                (self.output_size[0] - cropped_image.width) // 2,
                (self.output_size[1] - cropped_image.height) // 2
            ))

        return cropped_image

def BaseImageTransform(x_center = 400, y_center = 180, recenter_window = (180, 180), output_size = (50, 50)):
    return T.Compose([
        v2.RGB(),
        RecenterTransform(x_center, y_center, recenter_window),
        T.Resize(output_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def BaseVideoTransform(x_center = 400, y_center = 180, recenter_window = (180, 180), output_size = (50, 50), blur_sigma = 0):
    if blur_sigma == 0:
        return T.Compose([
            RecenterTransform(x_center, y_center, recenter_window),
            T.Resize(output_size),
            T.ToTensor(),
            T.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989])
        ])
    else:
        kernel_x = int(output_size[0]/8)
        kernel_y = int(output_size[1]/8)
        if kernel_x % 2 == 0:
             kernel_x = kernel_x + 1
        if kernel_y % 2 == 0:
             kernel_y = kernel_y + 1  
            
        return T.Compose([
            RecenterTransform(x_center, y_center, recenter_window),
            T.Resize(output_size),
            T.GaussianBlur((kernel_x, kernel_y), sigma = blur_sigma),
            T.ToTensor(),
            T.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989])
        ])
        