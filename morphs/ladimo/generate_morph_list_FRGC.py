import os

p_list = ['P1','P2']
gender_list = ['Female','Male']
source_dir = '../../TBIOM2020/TBIOM_copy/'
target_dir = './TBIOM_FRGC/'
ref_dir = '../../TBIOM2020/TBIOM_copy/Digital/Morph/MIPGAN_I/'



for part in p_list:
    source_part_dir = os.path.join(source_dir,'bonafide_contributed',part)

    ref_path = os.path.join(ref_dir,'ICAO_'+part)
    ref_list = os.listdir(ref_path)

    for gender in gender_list:
        rows = []
        source_part_gender_dir = os.path.join(source_part_dir,'Cropped_Resized_1024',gender)

        source_list = os.listdir(source_part_gender_dir)
        
        target_output_dir = os.path.join(target_dir,part,gender)
        os.makedirs(target_output_dir,exist_ok=True)

        subject_list =[]
        for source in source_list:
            subject_list.append(source[:-4])
        
        for ref_name in ref_list:
            splited_names = ref_name.split('-vs-')
            if splited_names[0] in subject_list:
                img1 = splited_names[0]+'.png'
                img2 = splited_names[1][:-4]+'.png'
                output = target_output_dir

                rows.append(img1+','+img2)       


        with open('Morph_LADIMO_'+part+'_'+gender+'.txt','a+') as f:
            for row in rows:
                f.write(row)
                f.write('\n')
