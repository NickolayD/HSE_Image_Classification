from numpy import array, random, hstack
import settings
from os import listdir, sep
from PIL import Image
from skimage import color
from skimage.feature import hog


class DataPreporation:

    def __init__(self):
        self.images = []
        self.labels = []
        self.hog_visualization = None

    def load_data(self, path, amount_per_category=10e6):
        ''' Загрузка данных из указанной директории.'''
        categories = listdir(path)
        for i in range(len(categories)):
            category_path = path + categories[i] + sep
            pics = listdir(category_path)
            for j in range(len(pics)):
                self.images.append(array(Image.open(category_path + pics[j]),
                                        dtype='uint8'
                                        )
                                  )
                self.labels.append(i)
                # выход из цикла в случае ограниченного считывания
                if j > amount_per_category:
                    break

    def convert_to_gray(self):
        '''
        Перевод массива чисел, соотв. цветному изображению, в массив числел,
        соответствующих серому изображению.
        '''
        for i in range(len(self.images)):
            self.images[i] = color.rgb2gray(self.images[i])

    def get_hog_features(self, visualize=False):
        '''
        Заменяет массив чисел, соответствующий серому изображению, на
        массив чисел, представляющих собой признаки изображения, полученные
        с помощью HOG.
        '''
        if visualize:
            self.hog_visualization = []
            for i in range(len(self.image)):
                hg, self.images[i] = hog(self.images[i],
                                         orientations=settings.orient,
                                         pixels_per_cells=settings.ppc,
                                         cells_per_block=settings.cpb,
                                         block_norm=settings.norm,
                                         visualize=True
                                         )
                self.hog_visualization.append(hg)
        else:
            for i in range(len(self.images)):
                self.images[i] = hog(self.images[i],
                                     orientations=settings.orient,
                                     pixels_per_cell=settings.ppc,
                                     cells_per_block=settings.cpb,
                                     block_norm=settings.norm
                                     )

    def shuffle(self):
        ''' Перемешивает датасет. '''
        data = hstack((array(self.images),
                       array(self.labels).reshape(len(self.labels), 1))
                      )
        random.shuffle(data)
        self.images, self.labels = data[:, :-1], data[:, -1:].ravel()

    def get_dataset(self):
        ''' Возвращает датасет '''
        return self.images, self.labels

    def get_hog_visualization(self):
        ''' Возвращает визцализацию HOG метода '''
        return self.hog_visualization
