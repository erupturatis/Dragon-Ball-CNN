import cv2
import os
import fnmatch
import shutil
import numpy as np
class ProcessImage(object):

    def __init__(self) -> None:
        pass
    
    def test(self):
        pass
    
    @staticmethod
    def image_to_numpy(img_path):
        image_list = list()
        i = 0
        for image in img_path:
            img = cv2.imread(image, cv2.IMREAD_UNCHANGED)
            # skipping black and white images
            if len(img.shape)!=3:
                continue

            img = img.T
            image_list.append(img)
            i+=1
            if i%100 == 0:
                print(f"{i} images processed")

        image_list = np.stack(image_list)
        return image_list



    @staticmethod
    def reshape_image(img_path, x:int = 128, y:int = 128):
        
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

        dim = (x, y)

        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        status = cv2.imwrite(img_path, resized)


    @staticmethod   
    def rescale_image(img_path, scale:float):
        if scale>100:
            scale = scale/100

        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        dim = (img.shape[0]*scale, img.shape[1]*scale)

        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        status = cv2.imwrite(img_path, resized)




class ProcessDataset(object):

    img_paths = list()
    labels = list()

    def __init__(self) -> None:
        pass

    def __call__(self, dataset_name, x:int, y:int) -> None:
        self.resize_dataset(dataset_name,x,y)
        

    def resize_dataset(self, dataset_name:str, x:int, y:int):
        self.clone_dataset(dataset_name)
        self.get_paths(f'{dataset_name}_processed')
        self.resize_path_images(x,y)


    def rescale_dataset(self, dataset_name:str, scale:float):
        self.clone_dataset(dataset_name)
        self.get_paths(f'{dataset_name}_processed')
        self.resize_path_images(scale)


    def rescale_path_images(self, scale):
        
        for image in self.img_paths:
            ProcessImage.rescale_image(image, scale=scale)


    def resize_path_images(self,x,y):
        i = 0
        for image in self.img_paths:
            ProcessImage.reshape_image(image,x,y)
            i+=1
            if i%100 == 0:
                print(f"{i} images processed")


    def get_paths(self, name:str):

        print("start getting paths")
        num_files = 0
        for path,dirs,files in os.walk(name):

            for filename in files:
                num_files += 1
                if filename.endswith(('.jpg', '.jpeg', '.png')):
                    fullname = os.path.abspath(os.path.join(path,filename))
                    self.img_paths.append(fullname)

                    if fullname.find("frieza")>=0:
                        self.labels.append(1)

                    elif fullname.find("gohan")>=0:
                        self.labels.append(2)

                    elif fullname.find("goku")>=0:
                        self.labels.append(3)

                    elif fullname.find("vegeta")>=0:
                        self.labels.append(4)
                    

        print(f"finished getting {num_files} paths")
        print("---------------------------------------------- \n\n")

    def get_dataset_as_numpy(self, name:str):
        self.get_paths(name)
        return ProcessImage.image_to_numpy(self.img_paths), self.labels


    @staticmethod
    def clone_dataset(dataset_name:str):

        from_directory = dataset_name
        to_directory = f"{dataset_name}_processed"
        print("start clone")
        shutil.copytree(from_directory, to_directory)
        print("cloned")


if __name__ == "__main__":
    pass