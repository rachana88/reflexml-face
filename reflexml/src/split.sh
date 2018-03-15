#!/bin/bash

PATH_TO_VID="$1"
TAR_PATH="$2"

TMPVID="$(mktemp -d 2>/dev/null || mktemp -t 'reflexml')".avi
PIC_FOLDER="$(mktemp -d 2>/dev/null || mktemp -d -t 'reflexml')"

echo "Ingesting video: $PATH_TO_VID"
echo "Saving to: $PIC_FOLDER"

echo "Splitting using ffmpeg"
ffmpeg -i "$PATH_TO_VID" -vf scale=iw/4:-1 -r 24 -y "$TMPVID" > /dev/null

mkdir -p "$PIC_FOLDER"

ffmpeg -i "$TMPVID" -y "$PIC_FOLDER"/pic%04d.jpg  > /dev/null


if [[ ! -z "$TAR_PATH" ]]; then
	ORIG="$(pwd)"
	cd "$PIC_FOLDER" || exit 1
	tar -cvzf "$ORIG"/"$TAR_PATH" ./*.jpg
fi

rm "$TMPVID"




