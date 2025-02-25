import os



def creat_filelist(input_path, classes):

    dir_image1 = []
    file_list = []
    for index, name in enumerate(classes):
        print('index', index)
        index_str = str(index)
        dir_image1_temp = input_path + '/' + name + '/'
        for dir2 in os.listdir(dir_image1_temp):
            dir_image2_temp = dir_image1_temp + '/' + dir2 + ' ' + index_str
            # dir_image2_temp1 = dir_image2_temp.join(' ')
            # dir_image2_temp2 = dir_image2_temp.join(index)
            file_list.append(dir_image2_temp)

    return dir_image1, file_list


def creat_txtfile(output_path, file_list):
    with open(output_path, 'w') as f:
        for list in file_list:
            print(list)
            f.write(str(list) + '\n')


def main():
    dir_image0 = 'dataset/train'
    dir_image1 = os.listdir(dir_image0)
    classes = dir_image1
    print(classes)
    dir_list, file_list = creat_filelist(dir_image0, classes)

    # print(file_list[0:3])
    output_path = 'train.txt'
    creat_txtfile(output_path, file_list)
    # print(output_path)




if __name__ == '__main__':
    main()