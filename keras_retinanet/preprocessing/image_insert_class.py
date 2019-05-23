import numpy as np
import imgaug as ia
import cv2
import random


class ImageInserter:
    
    def __init__(self, inserted_image_paths, size_range, angle_range, p=1.0,
                 feathering=False, thermal=False, shuffle_colors=True):
        self.p = p
        self.size_range = size_range
        self.angle_range = angle_range
        self.thermal = thermal
        self.shuffle_colors = shuffle_colors
        self._load_inserted_images(inserted_image_paths, feathering)
    
    
    def insert_images(self, images):
        num_images = len(images)
        insert_mask, size_samples, angle_samples, image_samples = self._generate_random_parameters(num_images)
        
        images = np.copy(images)
        # add empty bounding boxes for images without drones
        bboxes = [ia.BoundingBoxesOnImage([], img.shape) if not mask else None
                  for img, mask in zip(images, insert_mask)]
        
        insert_idx = np.arange(num_images)[insert_mask]
        for i, size, angle, inserted_image in zip(insert_idx, size_samples, angle_samples, image_samples):
            images[i], bbox = self._transform_and_insert(images[i], size, angle, inserted_image)
            bboxes[i] = ia.BoundingBoxesOnImage([bbox], images[i].shape)
                
        return images, bboxes
    
    
    def _load_inserted_images(self, paths, feathering):
        self.inserted_images = []
        for path in paths:
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
            if feathering:
                img = self._feather_image(img)
            self.inserted_images.append(img)
        
    
    def _generate_random_parameters(self, num_samples):
        insert_mask = (np.random.binomial(1, self.p, size=num_samples) == 1)
        num_inserted = np.sum(insert_mask)
        
        size_samples = np.exp(np.random.uniform(np.log(self.size_range[0]), np.log(self.size_range[1]),
                                                num_inserted))
        angle_samples = np.random.uniform(self.angle_range[0], self.angle_range[1], num_inserted)
        image_samples = random.choices(self.inserted_images, k=num_inserted)
        
        return insert_mask, size_samples, angle_samples, image_samples
    
    
    def _rotate_and_trim_image(self, image, angle):
        rows, cols = image.shape[0], image.shape[1]
        M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
        abs_cos = abs(M[0,0])
        abs_sin = abs(M[0,1])
        cols_rot = int(rows * abs_sin + cols * abs_cos)
        rows_rot = int(rows * abs_cos + cols * abs_sin)
        M[0, 2] += (cols_rot - cols) / 2
        M[1, 2] += (rows_rot - rows) / 2
        
        result = cv2.warpAffine(image, M, (cols_rot, rows_rot))
        x, y, w, h = cv2.boundingRect(result[:, :, 3])
        
        return result[y:y+h, x:x+w]
    
    
    def _feather_image(self, image):
        mask = image[:, :, 3]
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.GaussianBlur(mask, (5,5), 0)
        result = np.copy(image)
        result[:, :, 3] = mask
        return result

    
    def _transform_and_insert(self, background, size, angle, inserted_image):
        
        inserted_image = self._rotate_and_trim_image(inserted_image, angle)
        
        if self.thermal:
            inserted_image[..., :3] = 255 - inserted_image[..., :3]
            inserted_image[..., :3] = np.uint8(np.clip(inserted_image[..., :3] * np.random.uniform(1.0, 1.5), 0, 255))
        
        if self.shuffle_colors:
            perm = np.random.permutation(3)
            perm = np.append(perm, 3)
            inserted_image = inserted_image[:, :, perm]
        
        size = size * min(background.shape[0], background.shape[1])
        scale = size / max(inserted_image.shape)
        new_w = int(inserted_image.shape[1] * scale)
        new_h = int(inserted_image.shape[0] * scale)
        if new_h == 0 or new_w == 0:
            return background, None
        inserted_image = cv2.resize(inserted_image, (new_w, new_h))
        
        x1 = np.random.randint(background.shape[1] - inserted_image.shape[1])
        y1 = np.random.randint(background.shape[0] - inserted_image.shape[0])
        x2 = int(x1 + inserted_image.shape[1])
        y2 = int(y1 + inserted_image.shape[0])
        bbox = ia.BoundingBox(x1, y1, x2, y2)
                
        return self._insert_image(background, inserted_image, x1, y1), bbox
    
    
    def _insert_image(self, background, image, x1, y1):        
        mask = image[:, :, 3] / 255
        mask = np.stack((mask, mask, mask), axis=2)
        x2 = x1 + image.shape[1]
        y2 = y1 + image.shape[0]
        background[y1:y2, x1:x2] = np.uint8(background[y1:y2, x1:x2] * (1 - mask) + mask * image[:, :, :3])
        blured = cv2.GaussianBlur(background[y1:y2, x1:x2], (3,3), 0)
        background[y1:y2, x1:x2] = np.uint8(background[y1:y2, x1:x2] * (1 - mask) + mask * blured)

        return background