#!/bin/bash
# AI-Hub 파일 하나를 받아 풀고 분할 파일(.part0, .part1 ...)을 병합한다.
# 사용: aihub_download.sh <데이터셋번호> <파일키> <출력폴더>
# 인증 키는 환경변수 AIHUB_APIKEY 에서 읽는다(코드·로그에 쓰지 않는다).
# 주의: 서버가 이어받기(Range)를 지원하지 않아 중간에 끊기면 처음부터 다시 받는다.
set -euo pipefail
ds=$1; fk=$2; out=$3
mkdir -p "$out" && cd "$out"
tar_f="dl_$fk.tar"

curl -fL --retry 3 -H "apikey:$AIHUB_APIKEY" -o "$tar_f" \
  "https://api.aihub.or.kr/down/0.6/$ds.do?fileSn=$fk"
tar -xf "$tar_f"

# 같은 이름의 .partN 조각을 번호 순으로 이어 붙이고 조각을 지운다
find . -name '*.part0' -print0 | while IFS= read -r -d '' p; do
  base=${p%.part0}
  find "$(dirname "$p")" -name "$(basename "$base").part*" -print0 | sort -zV | xargs -0 cat > "$base"
  find "$(dirname "$p")" -name "$(basename "$base").part*" -delete
done
rm "$tar_f"
echo "완료: 파일키 $fk $(date '+%F %T')"
