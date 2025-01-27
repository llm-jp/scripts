import argparse
import dataclasses
import json
import logging
import pathlib
from typing import Optional, Any


@dataclasses.dataclass
class SpeechRecord:
    speechID: str
    speechOrder: int
    speaker: str
    speakerYomi: Optional[str]
    speakerGroup: Optional[str]
    speakerPosition: Optional[str]
    speakerRole: Optional[str]
    speech: str
    startPage: int
    createTime: str
    updateTime: str
    speechURL: str


@dataclasses.dataclass
class MeetingRecord:
    issueID: str
    imageKind: str
    searchObject: int
    session: int
    nameOfHouse: str
    nameOfMeeting: str
    issue: str
    date: str
    closing: Any
    speechRecord: list[SpeechRecord]
    meetingURL: str
    pdfURL: str

    def __post_init__(self):
        self.speechRecord = [SpeechRecord(**speech) for speech in self.speechRecord]


@dataclasses.dataclass
class KokkaiGiji:
    numberOfRecords: int
    numberOfReturn: int
    startRecord: int
    nextRecordPosition: Optional[int]
    meetingRecord: list[MeetingRecord]

    def __post_init__(self):
        self.meetingRecord = [MeetingRecord(**meeting) for meeting in self.meetingRecord]


def meeting_to_text(meeting: MeetingRecord) -> str:
    text = ""
    for speech in meeting.speechRecord:
        text += speech.speech.replace("\r\n", "\n").strip() + "\n\n"
    return text.strip()


def main() -> None:
    parser = argparse.ArgumentParser("Remove intra-sentence line breaks from text.")
    parser.add_argument("--input-dir", type=str, required=True, help="Input directory.")
    parser.add_argument("--output-file", type=str, required=True, help="Output file.")
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite output file."
    )
    args = parser.parse_args()

    input_dir = pathlib.Path(args.input_dir)
    file_paths = sorted(input_dir.glob("**/*.json"))

    output_file = pathlib.Path(args.output_file)

    with output_file.open("wt", encoding="utf-8") as fout:
        for file_path in file_paths:
            with file_path.open("rt", encoding="utf-8") as fin:
                dat = KokkaiGiji(**json.load(fin))
            
            for meeting in dat.meetingRecord:
                instance = {
                    "text": meeting_to_text(meeting),
                    "meta": {
                        "issueID": meeting.issueID,
                        "imageKind": meeting.imageKind,
                        "searchObject": meeting.searchObject,
                        "session": meeting.session,
                        "nameOfHouse": meeting.nameOfHouse,
                        "nameOfMeeting": meeting.nameOfMeeting,
                        "issue": meeting.issue,
                        "date": meeting.date,
                        "closing": meeting.closing,
                        "meetingURL": meeting.meetingURL,
                        "pdfURL": meeting.pdfURL,
                    },
                }
                fout.write(json.dumps(instance, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )
    main()
