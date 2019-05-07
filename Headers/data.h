#ifndef data_h
#define data_h

#include <stdio.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <Eigen/Dense>

using namespace Eigen;

int ReverseInt (int i) //Still not entirely sure of the purpose of this.
{
    unsigned char ch1, ch2, ch3, ch4;
    ch1 = i & 255;
    ch2 = (i >> 8) & 255;
    ch3 = (i >> 16) & 255;
    ch4 = (i >> 24) & 255;
    return((int) ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}


void read_images(std::string filename, Matrix<unsigned char,Dynamic,Dynamic> & X){
    std::ifstream file (filename, std::ios::binary);
    if (file.is_open())
    {
        int magic_number = 0;
        int number_of_images = 0;
        int n_rows = 0;
        int n_cols = 0;

        file.read((char*) &magic_number, sizeof(magic_number));
        magic_number = ReverseInt(magic_number);
        file.read((char*) &number_of_images,sizeof(number_of_images));
        number_of_images = ReverseInt(number_of_images);
        file.read((char*) &n_rows, sizeof(n_rows));
        n_rows = ReverseInt(n_rows);
        file.read((char*) &n_cols, sizeof(n_cols));
        n_cols = ReverseInt(n_cols);

        int image_size = n_rows*n_cols;

        X.resize(image_size,number_of_images);

        for(int i=0; i < number_of_images; i++){
            for(int j = 0; j < image_size; ++j){
                unsigned char temp = 0;
                file.read((char*) &temp, sizeof(temp));
                X(j,i) = temp;
            }
        }
    }
}


void read_label(std::string filename, std::vector<int> & Y){

    std::ifstream file (filename, std::ios::binary);
    if (file.is_open()){

        int magic_number = 0;
        int number_of_images = 0;

        file.read((char*) &magic_number, sizeof(magic_number));
        magic_number = ReverseInt(magic_number);
        file.read((char*) &number_of_images,sizeof(number_of_images));
        number_of_images = ReverseInt(number_of_images);

        if (!Y.empty()) Y.clear();
        Y.resize(number_of_images);

        for(unsigned i = 0; i < number_of_images; ++i){
            unsigned char temp = 0;
            file.read((char*) &temp, sizeof(temp));
            Y[i] = int(temp);
        }
    }
}
#endif /* data_h */
