import json

from src.tasks.common.multimodal_structured import MultimodalStructuredHandler


class DummyMultimodalStructuredHandler(MultimodalStructuredHandler):
    name = "dummy_multimodal_structured"

    def _load_samples(self):
        return []


def test_load_reads_dataset_metrics_config_for_multimodal_structured(tmp_path):
    schema = {
        "type": "object",
        "properties": {
            "cronograma": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "fecha": {"type": "string"},
                        "hora": {"type": "string"},
                    },
                },
            }
        },
    }
    metrics_config = {
        "unordered_arrays": {
            "cronograma": {
                "key_fields": ["fecha", "hora"],
            }
        }
    }

    (tmp_path / "schema.json").write_text(json.dumps(schema), encoding="utf-8")
    (tmp_path / "metrics_config.json").write_text(json.dumps(metrics_config), encoding="utf-8")

    handler = DummyMultimodalStructuredHandler()
    handler.data_dir = tmp_path

    handler.load()

    assert handler.dataset_metrics_config == metrics_config
    assert handler.metrics_calculator.config["metrics"]["unordered_arrays"] == metrics_config["unordered_arrays"]
