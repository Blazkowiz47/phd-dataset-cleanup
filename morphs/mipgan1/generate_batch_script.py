#__author__ = "Haoyu Zhang"
#__copyright__ = "Copyright (C) 2021 Norwegian University of Science and Technology"
#__license__ = "License Agreement provided by Norwegian University of Science and Technology (NTNU)" \
#              "(MIPGAN-license-210420.pdf)"
#__version__ = "1.0"

import csv
import argparse
import os
def main():
    parser = argparse.ArgumentParser(description='Split morph list and generate a bash to run it', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--system', default='Ubuntu', help='The system you are using, will generate .sh file for Ubuntu or .bat file for Windows')
    parser.add_argument('--morph_list_dir', default='trainMale_list.csv', help='Directory of the morphing list' )
    parser.add_argument('--output_dir', default='splited_morph_list', help='Directory for storing splited morphs' )
    parser.add_argument('--split_interval', default='50', help='Number of morphs per splited morph_list', type=int )

    args, other_args = parser.parse_known_args()

    morph_list_CSV = args.morph_list_dir
    reset_step = args.split_interval
    ref_images=[]
    with open(morph_list_CSV, newline='') as csvfile:
        print('reading data: ' + morph_list_CSV)
        line = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in line:
            if row:
                ref_images.append(row)

    print(len(ref_images))

    splited_lists = [ref_images[i:i+reset_step] for i in range(0, len(ref_images), reset_step)]

    output_dir = args.output_dir
    os.makedirs(output_dir,exist_ok=True)

    cnt=0
    morph_list_names=[]
    for splited_list in splited_lists:
        morph_list_names.append(os.path.join(output_dir,'morph_list_splited_'+str(cnt)+'.csv'))
        f = open(os.path.join(output_dir,'morph_list_splited_'+str(cnt)+'.csv'),'w',encoding='utf-8')
        csv_writer = csv.writer(f)
        for row in splited_list:
            csv_writer.writerow(row)
        f.close
        cnt=cnt+1
    if args.system=='Ubuntu':
        f = open('ubuntu_MIPGAN_script.sh','w',encoding='utf-8')
        f.write('#! /bin/bash\n')
        for morph_list_name in morph_list_names:
            f.write('python encode_images_MIPGAN.py %s aligned_images generated_images latent_representations\n'%morph_list_name)
        f.close
    elif args.system=='Windows':
        f = open('ubuntu_MIPGAN_script.bat','w',encoding='utf-8')
        for morph_list_name in morph_list_names:
            f.write('python encode_images_MIPGAN.py %s aligned_images generated_images latent_representations\n'%morph_list_name)
        f.close


if __name__ == "__main__":
    main()