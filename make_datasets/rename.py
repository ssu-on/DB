import os
import re

def rename_files_in_directory(directory_path):
    """
    지정된 디렉토리 내의 파일 중 'moive3_숫자.jpg' 패턴을
    'movie3_숫자.jpg' 패턴으로 변경합니다.
    """
    
    # 1. 입력 경로 유효성 검사
    if not os.path.isdir(directory_path):
        print(f"오류: '{directory_path}'는 유효한 디렉토리가 아닙니다.")
        return

    # 변경된 파일 카운트
    renamed_count = 0
    
    # 파일명 검색을 위한 정규 표현식 패턴: 'moive3_'로 시작하고 그 뒤에 숫자(.jpg 이전)가 오는 경우
    # re.IGNORECASE는 대소문자 구분을 무시합니다.
    pattern = re.compile(r"moive3_(\d+)\.jpg$", re.IGNORECASE)

    print(f"'{directory_path}' 디렉토리 내에서 파일명 변경을 시작합니다...")

    # 2. 디렉토리 내 모든 파일 순회
    for filename in os.listdir(directory_path):
        
        # 정규 표현식 매칭 시도
        match = pattern.match(filename)
        
        if match:
            # 3. 새로운 파일명 생성
            # match.group(1)은 정규식에서 괄호(\d+)로 캡처된 '숫자' 부분입니다.
            new_filename = f"movie3_{match.group(1)}.jpg"
            
            # 4. 전체 경로 설정 (원본 및 대상)
            old_path = os.path.join(directory_path, filename)
            new_path = os.path.join(directory_path, new_filename)
            
            # 파일 이름 변경 실행
            try:
                os.rename(old_path, new_path)
                print(f"✅ 변경 완료: {filename} -> {new_filename}")
                renamed_count += 1
            except Exception as e:
                print(f"❌ 변경 실패: {filename} -> {new_filename}. 오류: {e}")
        # else:
            # 패턴에 맞지 않는 파일은 건너뜁니다.

    print("-" * 30)
    print(f"총 {renamed_count}개의 파일명이 성공적으로 변경되었습니다.")
    if renamed_count == 0:
        print("변경할 파일(moive3_숫자.jpg)이 없거나 이미 변경되었습니다.")


if __name__ == "__main__":
    # 사용자로부터 디렉토리 경로 입력받기
    input_dir = input("파일명을 변경할 폴더의 경로를 입력해 주세요 (예: C:\\Users\\User\\Desktop\\data): ")
    input_dir = '/data/subtitle/DBNet/raw_data/movie3'
    # 입력받은 경로로 함수 실행
    rename_files_in_directory(input_dir.strip())