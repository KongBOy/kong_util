from kong_util.build_dataset_combine import Check_dir_exist_and_build
from kong_util.util import get_dir_certain_file_names
def Convert_dir_pdf_to_jpg(ord_dir, dst_dir="pdf_to_jpg_result", print_msg=False):
    '''
    使用前的安裝：https://github.com/Belval/pdf2image
        1.  pip install pdf2image
        2a. 去 https://github.com/oschwartz10612/poppler-windows/releases/ 下載 Release-21.09.0.zip
        2b. zip解壓縮，poppler-21.09.0/Library/bin 資料夾加入 環境變數 Path中(建議解壓縮下來的東西放在C槽下，因為加入Path裡的路徑不能有中文，放C槽下最保險)
    '''
    from pdf2image import convert_from_path
    Check_dir_exist_and_build(dst_dir)

    file_names = get_dir_certain_file_names(ord_dir, certain_word=".pdf")  ### 取得dir 內的 .pdf
    for file_name in file_names:
        ord_pdf_path = f"{ord_dir}/{file_name}"        ### 定位出 來源pdf 的 path
        dst_half_path = f"{dst_dir}/{file_name[:-4]}"  ### 因為後面還要接上頁碼，所以這邊定位出 不含副檔名的 位置

        pages = convert_from_path(ord_pdf_path, dpi=300)  ### pdf -> jpg

        for go_page, page in enumerate(pages):  ### 走訪每頁
            dst_full_path = "%s_%03i.jpg" % (dst_half_path, go_page + 1)  ### 根據 頁碼 定位出 精確 dst_path
            page.save(dst_full_path, "JPEG")
        if(print_msg): print(f"{ord_pdf_path} -> jpg finish")