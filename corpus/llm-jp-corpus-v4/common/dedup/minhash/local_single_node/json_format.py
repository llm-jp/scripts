import orjson


def default_adapter(self, data: dict, path: str, id_in_file: int | str):
    """
    The default data adapter to adapt input data into the datatrove Document format

    Args:
        data: a dictionary with the "raw" representation of the data
        path: file path or source for this sample
        id_in_file: its id in this particular file or source

    Returns: a dictionary with text, id, media and metadata fields

    """
    return {
        "text": data.pop(self.text_key, ""),
        "id": data.pop(self.id_key, f"{path}/{id_in_file}"),
        "media": data.pop("media", []),
        # remaining data goes into metadata
        "metadata": data.pop("metadata", {}) | data,
    }


def custom_adapter(
        self,
        data: dict,
        path: str,
        id_in_file: int | str):
    """
    カスタムアダプタの実装例
    「学術文献コーパス」の場合、
    - id は "id" または "meta/database + meta/id"
    - metadata は "meta" 内のオブジェクト
    - text は "text" から読み込む
    """
    metadata = data.get("meta", {})
    if isinstance(metadata, str):
        try:
            metadata = orjson.loads(metadata)
        except orjson.JSONDecodeError:
            pass
    if not isinstance(metadata, dict):
        metadata = {"metadata": metadata}

    if "id" in data:
        doc_id = data.get("id")
    elif "database" in metadata and "id" in metadata:
        doc_id = metadata["database"] + ":" + metadata["id"]
    else:
        doc_id = f"{path}/{id_in_file}"

    text = data.get("text", "")
    if text == "":
        logger.warning(f"Empty text in document with id={doc_id}")

    return {
        "text": text,
        "id": doc_id,
        "media": data.get("media", []),
        "metadata": metadata
    }


adapter = default_adapter
