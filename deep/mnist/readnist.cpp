#include "readnist.hpp"


int reverseInt (int i)
{
    unsigned char ch1, ch2, ch3, ch4;
    ch1 = i & 255;
    ch2 = (i >> 8) & 255;
    ch3 = (i >> 16) & 255;
    ch4 = (i >> 24) & 255;
    return((int) ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}



std::vector<Pattern> read_Mnist(std::string filenameImages, std::string filenameLabels, int maxImages)
{
    std::vector<Pattern> vec;
    std::ifstream fileImages(filenameImages, std::ios::binary);
    std::ifstream fileLabels(filenameLabels, std::ios::binary);
    if (fileImages.is_open () && fileLabels.is_open ())
    {
        int magic_number = 0;
        int number_of_images = 0;
        int n_rows = 0;
        int n_cols = 0;
        fileImages.read((char*) &magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);
        fileImages.read((char*) &number_of_images,sizeof(number_of_images));
        number_of_images = reverseInt(number_of_images);

        if (maxImages > 0)
            number_of_images = maxImages;

        fileImages.read((char*) &n_rows, sizeof(n_rows));
        n_rows = reverseInt(n_rows);
        fileImages.read((char*) &n_cols, sizeof(n_cols));
        n_cols = reverseInt(n_cols);
        for(int i = 0; i < number_of_images; ++i)
        {
            Pattern tmpPattern;
            for(int r = 0; r < n_rows; ++r)
            {
                for(int c = 0; c < n_cols; ++c)
                {
                    unsigned char tempPixel = 0;
                    fileImages.read((char*) &tempPixel, sizeof(tempPixel));
                    //tpmat.at<uchar>(r, c) = (int) temp;
                    double pixelValue = tempPixel/255.0;
                    tmpPattern.addInput (pixelValue);
                }
            }

            unsigned char tempLabel = 0;
            fileLabels.read((char*) &tempLabel, sizeof(tempLabel));
            for (int i = 0; i < 10; ++i)
            {
                tmpPattern.addOutput ((double)int(tempLabel==i));
            }

            tmpPattern.weight (1.0);
            vec.push_back(tmpPattern);
        }
    }
    std::cout << "read MNIST data: " << vec.size () << std::endl;
    return vec;
}

