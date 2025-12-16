import re
import os

def correct_filenames_in_txt(input_filepath, output_filepath=None):
    """
    텍스트 파일 내의 'moive' 오타를 'movie'로 수정합니다.

    :param input_filepath: 원본 TXT 파일 경로
    :param output_filepath: 변경된 내용을 저장할 새로운 TXT 파일 경로 (None이면 원본 파일명에 '_fixed' 추가)
    """

    # 1. 출력 파일 경로 설정
    if output_filepath is None:
        # 원본 파일명.txt -> 원본 파일명_fixed.txt
        base, ext = os.path.splitext(input_filepath)
        output_filepath = base + "_fixed" + ext
    
    # 변경된 줄 카운트
    corrected_lines_count = 0
    
    # 파일명 패턴 정의: 'moive'를 'movie'로 변경
    # re.sub()를 사용할 것이므로, 변경할 문자열('moive')만 지정
    search_pattern = r"moive"
    replace_pattern = r"movie"

    try:
        # 2. 원본 파일 읽기
        with open(input_filepath, 'r', encoding='utf-8') as infile:
            lines = infile.readlines()
        
        corrected_lines = []

        # 3. 각 줄을 순회하며 변경 작업 수행
        for line in lines:
            # 줄의 앞뒤 공백 및 줄 바꿈 문자 제거
            stripped_line = line.strip()
            
            # 정규 표현식을 사용하여 'moive'를 'movie'로 치환
            # (re.IGNORECASE를 사용하면 대소문자 관계없이 'Moive'도 'Movie'로 변경됨)
            corrected_line = re.sub(search_pattern, replace_pattern, stripped_line, flags=re.IGNORECASE)
            
            # 변경된 줄 목록에 추가 (줄 바꿈 문자 추가)
            corrected_lines.append(corrected_line + '\n')
            
            # 변경 여부 확인 및 카운트
            if corrected_line != stripped_line:
                corrected_lines_count += 1
                
        # 4. 새로운 파일에 변경된 내용 쓰기
        with open(output_filepath, 'w', encoding='utf-8') as outfile:
            outfile.writelines(corrected_lines)
            
        print("-" * 40)
        print(f"✅ 파일 내용 변경 완료!")
        print(f"원본 파일: {input_filepath}")
        print(f"저장된 파일: {output_filepath}")
        print(f"총 {corrected_lines_count}개의 줄이 성공적으로 변경되었습니다.")
        print("-" * 40)
        
    except FileNotFoundError:
        print(f"❌ 오류: '{input_filepath}' 파일을 찾을 수 없습니다.")
    except Exception as e:
        print(f"❌ 오류 발생: {e}")

if __name__ == "__main__":
    # 사용자로부터 TXT 파일 경로 입력받기
    #input_file = input("내용을 변경할 TXT 파일의 경로를 입력해 주세요: ").strip()
    input_file = '/data/subtitle/DBNet/raw_data/moive3.txt'
    if not input_file:
        print("파일 경로가 입력되지 않았습니다. 종료합니다.")
    else:
        # 함수 실행 (새로운 파일 경로를 지정하지 않으면 원본 파일명_fixed.txt로 저장됨)
        correct_filenames_in_txt(input_file)