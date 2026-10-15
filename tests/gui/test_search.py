import yaml


def load(backend, yaml_file):
    assert backend.load_yaml(str(yaml_file))["success"]


def ids(result):
    return [match["id"] for match in result["matches"]]


class TestSearchArtefacts:
    def test_no_page_loaded(self, backend_with_extension):
        result = backend_with_extension.search_artefacts("hello")

        assert result["success"] is False
        assert "No graph loaded" in result["error"]

    def test_finds_text_inside_an_artefact_ignoring_case(
        self, backend_with_extension, temp_yaml_file
    ):
        load(backend_with_extension, temp_yaml_file)

        result = backend_with_extension.search_artefacts("hello")

        assert result["success"] is True
        assert ids(result) == ["text_001"]
        match = result["matches"][0]
        assert match["field"] == "text"
        assert match["label"] == "text_label"
        assert match["type"] == "collectra.Text"
        assert "Hello World" in match["snippet"]

    def test_does_not_search_dag_labels_or_ids(
        self, backend_with_extension, temp_yaml_file
    ):
        load(backend_with_extension, temp_yaml_file)

        # "crop" and "text_001" only occur in a label / an id, never in content.
        assert backend_with_extension.search_artefacts("crop")["matches"] == []
        assert backend_with_extension.search_artefacts("text_001")["matches"] == []
        assert backend_with_extension.search_artefacts("_label")["matches"] == []

    def test_finds_the_custom_name_of_an_artefact(
        self, backend_with_extension, tmp_path
    ):
        folder = tmp_path / "page.collectra"
        folder.mkdir()
        (folder / "results.yaml").write_text(
            yaml.dump(
                {
                    "stamp": [
                        {
                            "type": "collectra.ImageCrop",
                            "id": "stamp_001",
                            "name": "Red Stamp",
                            "data": "x.jpg",
                            "x_center": 0.5,
                            "y_center": 0.5,
                            "width_relative": 0.1,
                            "height_relative": 0.1,
                        }
                    ]
                }
            )
        )
        load(backend_with_extension, folder / "results.yaml")

        result = backend_with_extension.search_artefacts("red stamp")

        assert ids(result) == ["stamp_001"]
        assert result["matches"][0]["field"] == "name"

    def test_each_artefact_matches_once_in_page_order(
        self, backend_with_extension, tmp_path
    ):
        folder = tmp_path / "page.collectra"
        folder.mkdir()
        (folder / "results.yaml").write_text(
            yaml.dump(
                {
                    "top": [
                        {
                            "type": "collectra.Image",
                            "id": "i1",
                            "name": "alpha one",
                            "data": "x.jpg",
                        }
                    ],
                    "mid": [
                        {
                            "type": "collectra.ImageCrop",
                            "id": "c1",
                            "parents": "i1",
                            "name": "alpha two",
                            "data": "x.jpg",
                            "x_center": 0.5,
                            "y_center": 0.5,
                            "width_relative": 0.1,
                            "height_relative": 0.1,
                        }
                    ],
                    "low": [
                        {
                            "type": "collectra.Text",
                            "id": "t1",
                            "parents": "c1",
                            "data": "alpha three",
                        }
                    ],
                }
            )
        )
        load(backend_with_extension, folder / "results.yaml")

        result = backend_with_extension.search_artefacts("alpha")

        assert ids(result) == ["i1", "c1", "t1"]

    def test_a_query_that_matches_nothing_returns_no_matches(
        self, backend_with_extension, temp_yaml_file
    ):
        load(backend_with_extension, temp_yaml_file)

        assert backend_with_extension.search_artefacts("zzzz") == {
            "success": True,
            "matches": [],
        }

    def test_a_blank_query_returns_no_matches(
        self, backend_with_extension, temp_yaml_file
    ):
        load(backend_with_extension, temp_yaml_file)

        for query in ["", "   ", None]:
            assert backend_with_extension.search_artefacts(query)["matches"] == []

    def test_text_comes_before_name_when_both_match(
        self, backend_with_extension, tmp_path
    ):
        folder = tmp_path / "page.collectra"
        folder.mkdir()
        (folder / "results.yaml").write_text(
            yaml.dump(
                {
                    "note": [
                        {
                            "type": "collectra.Text",
                            "id": "note_001",
                            "name": "needle name",
                            "data": "a needle in the text",
                        }
                    ]
                }
            )
        )
        load(backend_with_extension, folder / "results.yaml")

        match = backend_with_extension.search_artefacts("needle")["matches"][0]

        assert match["field"] == "text"

    def test_reads_text_that_lives_in_a_file(self, backend_with_extension, tmp_path):
        folder = tmp_path / "page.collectra"
        folder.mkdir()
        (folder / "transcription.md").write_text("The quick brown fox")
        (folder / "results.yaml").write_text(
            yaml.dump(
                {
                    "transcription": [
                        {
                            "type": "collectra.Text",
                            "id": "transcription_001",
                            "path": "transcription.md",
                        }
                    ]
                }
            )
        )
        load(backend_with_extension, folder / "results.yaml")

        result = backend_with_extension.search_artefacts("brown fox")

        assert ids(result) == ["transcription_001"]
        assert result["matches"][0]["field"] == "text"

    def test_snippet_is_trimmed_around_the_match_in_long_text(
        self, backend_with_extension, tmp_path
    ):
        folder = tmp_path / "page.collectra"
        folder.mkdir()
        long_text = "x" * 200 + " needle " + "y" * 200
        (folder / "results.yaml").write_text(
            yaml.dump(
                {"note": [{"type": "collectra.Text", "id": "n1", "data": long_text}]}
            )
        )
        load(backend_with_extension, folder / "results.yaml")

        snippet = backend_with_extension.search_artefacts("needle")["matches"][0][
            "snippet"
        ]

        assert "needle" in snippet
        assert len(snippet) < 120
        assert snippet.startswith("…") and snippet.endswith("…")

    def test_searching_changes_nothing(self, backend_with_extension, temp_yaml_file):
        load(backend_with_extension, temp_yaml_file)
        before = temp_yaml_file.read_text()

        backend_with_extension.search_artefacts("hello")

        assert temp_yaml_file.read_text() == before


class TestPipelineOrder:
    def test_ranks_follow_the_layers_of_the_dag_then_the_edge_order(self):
        from collectra.gui.pipeline_order import pipeline_label_ranks

        graph = {
            "nodes": [
                {"id": n} for n in ["det", "sheet", "pl", "hw", "orient", "field"]
            ],
            "edges": [
                {"source": "sheet", "target": "det"},
                {"source": "det", "target": "pl"},
                {"source": "det", "target": "hw"},
                {"source": "pl", "target": "orient"},
                {"source": "orient", "target": "field"},
            ],
        }

        ranks = pipeline_label_ranks(graph)

        ordered = sorted(ranks, key=ranks.get)
        assert ordered == ["sheet", "det", "pl", "hw", "orient", "field"]

    def test_a_node_sits_below_its_longest_chain_of_parents(self):
        from collectra.gui.pipeline_order import pipeline_label_ranks

        graph = {
            "nodes": [{"id": n} for n in "abcd"],
            "edges": [
                {"source": "a", "target": "b"},
                {"source": "b", "target": "c"},
                {"source": "a", "target": "c"},
                {"source": "a", "target": "d"},
            ],
        }

        ranks = pipeline_label_ranks(graph)

        assert sorted(ranks, key=ranks.get) == ["a", "b", "d", "c"]

    def test_a_cycle_does_not_hang_and_every_node_gets_a_rank(self):
        from collectra.gui.pipeline_order import pipeline_label_ranks

        graph = {
            "nodes": [{"id": "a"}, {"id": "b"}],
            "edges": [{"source": "a", "target": "b"}, {"source": "b", "target": "a"}],
        }

        assert set(pipeline_label_ranks(graph)) == {"a", "b"}

    def test_matches_are_listed_in_pipeline_order_not_page_order(
        self, backend_with_extension, tmp_path, monkeypatch
    ):
        folder = tmp_path / "page.collectra"
        folder.mkdir()
        (folder / "results.yaml").write_text(
            yaml.dump(
                {
                    "first": [{"type": "collectra.Text", "id": "t1", "data": "alpha"}],
                    "second": [{"type": "collectra.Text", "id": "t2", "data": "alpha"}],
                    "third": [{"type": "collectra.Text", "id": "t3", "data": "alpha"}],
                }
            )
        )
        load(backend_with_extension, folder / "results.yaml")
        monkeypatch.setattr(
            backend_with_extension,
            "_pipeline_label_ranks",
            lambda: {"third": 0, "first": 1, "second": 2},
        )

        result = backend_with_extension.search_artefacts("alpha")

        assert ids(result) == ["t3", "t1", "t2"]

    def test_labels_the_pipeline_does_not_know_come_last_in_page_order(
        self, backend_with_extension, tmp_path, monkeypatch
    ):
        folder = tmp_path / "page.collectra"
        folder.mkdir()
        (folder / "results.yaml").write_text(
            yaml.dump(
                {
                    "made_by_hand": [
                        {"type": "collectra.Text", "id": "h1", "data": "alpha"}
                    ],
                    "known": [{"type": "collectra.Text", "id": "k1", "data": "alpha"}],
                }
            )
        )
        load(backend_with_extension, folder / "results.yaml")
        monkeypatch.setattr(
            backend_with_extension, "_pipeline_label_ranks", lambda: {"known": 0}
        )

        assert ids(backend_with_extension.search_artefacts("alpha")) == ["k1", "h1"]

    def test_the_pipeline_being_unreadable_keeps_page_order(
        self, backend_with_extension, temp_yaml_file
    ):
        load(backend_with_extension, temp_yaml_file)

        # The test pipeline defines no graph; searching must still work.
        assert ids(backend_with_extension.search_artefacts("hello")) == ["text_001"]


class TestSearchRobustness:
    def make(self, backend, tmp_path, records, files=None):
        folder = tmp_path / "page.collectra"
        folder.mkdir()
        for name, content in (files or {}).items():
            (folder / name).write_text(content)
        (folder / "results.yaml").write_text(yaml.dump(records))
        load(backend, folder / "results.yaml")

    def test_a_missing_text_file_is_not_searched_as_if_its_path_were_the_text(
        self, backend_with_extension, tmp_path
    ):
        self.make(
            backend_with_extension,
            tmp_path,
            {
                "note": [
                    {"type": "collectra.Text", "id": "n1", "path": "gone-secret.md"}
                ]
            },
        )

        assert backend_with_extension.search_artefacts("secret")["matches"] == []

    def test_one_unreadable_artefact_does_not_stop_the_search(
        self, backend_with_extension, tmp_path
    ):
        self.make(
            backend_with_extension,
            tmp_path,
            {
                "bad": [
                    {"type": "collectra.NoSuchType", "id": "b1", "name": "needle bad"}
                ],
                "good": [
                    {"type": "collectra.Text", "id": "g1", "data": "a needle here"}
                ],
            },
        )

        result = backend_with_extension.search_artefacts("needle")

        assert "g1" in ids(result)

    def test_a_binary_text_file_does_not_stop_the_search(
        self, backend_with_extension, tmp_path
    ):
        folder = tmp_path / "page.collectra"
        folder.mkdir()
        (folder / "blob.md").write_bytes(b"\xff\xfe\x00\xd8 not utf-8")
        (folder / "results.yaml").write_text(
            yaml.dump(
                {
                    "blob": [{"type": "collectra.Text", "id": "t0", "path": "blob.md"}],
                    "ok": [{"type": "collectra.Text", "id": "t1", "data": "needle"}],
                }
            )
        )
        load(backend_with_extension, folder / "results.yaml")

        assert ids(backend_with_extension.search_artefacts("needle")) == ["t1"]

    def test_the_snippet_is_cut_from_the_right_place_when_lowercasing_changes_length(
        self, backend_with_extension, tmp_path
    ):
        text = "İ" * 30 + " needle " + "x" * 30
        self.make(
            backend_with_extension,
            tmp_path,
            {"note": [{"type": "collectra.Text", "id": "n1", "data": text}]},
        )

        snippet = backend_with_extension.search_artefacts("needle")["matches"][0][
            "snippet"
        ]

        assert "needle" in snippet
        # The match is near the start of the text, so nothing precedes the cut: a
        # position measured in the lower-cased copy would wrongly start further on.
        assert not snippet.startswith("…")
