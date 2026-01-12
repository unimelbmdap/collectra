# Progress Log

## 2026-01-12: IoU Grouping Feature (Task 3) - Verified ✅

**Status**: 91 tests passing, PRD marked as complete

**Feature**: Matching text content to bounding boxes using IoU-based grouping

**Functions Added to `/workspace/collectra/ensemble.py`**:
1. `calculate_iou(box1, box2)` - Calculates Intersection over Union between two bounding boxes
   - Converts center-format boxes to corner format (x1, y1, x2, y2)
   - Handles edge cases: zero-size and negative-size boxes return 0.0
   - Returns IoU value (0.0 to 1.0)

2. `calculate_centroid_box(boxes)` - Calculates average of all 4 fields across boxes
   - Raises ValueError for empty list
   - Returns centroid bounding box dict

3. `group_by_bounding_box(labels, iou_threshold=0.6)` - Main grouping function
   - Groups labels by bounding box similarity using IoU
   - Conflict resolution: when same source folder appears twice in a group, keeps entry with higher IoU to group centroid
   - Logs warnings for conflict resolutions
   - Returns list of groups: `[(text_content, bounding_box, source_folder_path), ...]`

**Algorithm**:
- For each label tuple, iterate through existing groups
- Calculate max IoU between current box and all boxes in group
- If max IoU > 0.6, add to group (with conflict resolution if same source exists)
- If no group matches, create new group

**Tests Added** (22 new tests):
- `TestCalculateIou` (8 tests): identical boxes, non-overlapping, partial overlap, edge cases
- `TestCalculateCentroidBox` (4 tests): single box, multiple boxes, empty list
- `TestGroupByBoundingBox` (10 tests): grouping, conflicts, threshold boundary, output format

**Next Steps**: Task 4 - Generate ensembled value for each group using edit distance

---

## 2026-01-12: Ensemble Feature Implementation (Task 2) - Verified ✅

**Status**: 69 tests passing, PRD marked as complete

**Feature**: Extract labels with bounding boxes from source collectra files

**Functions Added to `/workspace/collectra/ensemble.py`**:
1. `get_source_collectra_files(folder_name, link_yaml_path)` - Returns list of source collectra file paths from link.yaml for a given ensemble folder name
2. `load_results_yaml(collectra_folder)` - Loads and parses results.yaml from a collectra folder
3. `extract_labels_with_bounding_boxes(source_folders)` - Main function that extracts all labels with their text content and bounding boxes
4. `_get_bounding_box_from_entry(entry)` - Helper to extract bounding box dict from an ImageCrop entry
5. `_find_parent_with_bounding_box(parent_id, results_data)` - Helper that traverses parent chain to find last ImageCrop with bounding box

**Output Format**:
- Returns list of tuples: `(label_name, text_content, bounding_box_dict, source_folder_path)`
- Bounding box dict: `{x_center, y_center, width_relative, height_relative}`

**Key Implementation Details**:
- Handles both `collectra.Text` and `collectra.ImageCrop` types
- For Text types: traverses upward through parents to find last ImageCrop with bounding box
- Handles single parent (string) and multiple parents (list) - uses LAST parent
- Skips `collectra_results_metadata` entries
- Graceful handling of missing results.yaml files

**Tests Added** (25 new tests):
- `TestGetSourceCollectraFiles` (4 tests)
- `TestLoadResultsYaml` (3 tests)
- `TestGetBoundingBoxFromEntry` (3 tests)
- `TestFindParentWithBoundingBox` (6 tests)
- `TestExtractLabelsWithBoundingBoxes` (9 tests)

**Next Steps**: Task 3 - Matching text content to bounding boxes (IoU grouping)

---

## 2026-01-12: Ensemble File Creation Feature - Verified ✅

**Status**: All 44 tests passing, PRD marked as complete

**Verification Summary**:
- CLI command: `collectra ensemble <folder1> <folder2> ... --output <output_folder>` ✅
- Duplicate detection with ValueError ✅
- Missing file warnings logged ✅
- Empty results.yaml created ✅
- Source artifacts (images) copied from first source ✅
- link.yaml format matches PRD specification ✅

---

## 2026-01-12: Ensemble File Creation Feature (Task 1)

**Feature**: CLI command and module for ensembling collectra files from multiple sources

**Changes Made**:
1. Created `/workspace/collectra/ensemble.py` - Core ensemble module with:
   - `find_collectra_files()` - Find collectra folders by extension in a directory
   - `verify_collectra_files()` - Verify files exist across all input folders
   - `create_ensemble_output()` - Create output structure with empty results.yaml, copied image, and link.yaml
   - `ensemble_files()` - Main entry point combining all functions
   - `load_link_yaml()` / `get_ensemble_folder()` - Utilities for reading link.yaml

2. Added `ensemble` command to `/workspace/collectra/main.py`:
   - Usage: `collectra ensemble <folder1> <folder2> ... --output <output_folder>`
   - Supports `--extension` / `-e` for custom file extensions
   - Rich console output with file counts, warnings, and success messages

3. Created `/workspace/tests/unit/test_ensemble.py` with 44 unit tests covering:
   - File finding, verification, and output creation
   - Edge cases (duplicates, missing files, permissions)
   - link.yaml loading and folder retrieval

**link.yaml Format**:
```yaml
file1.grapto:
  - /path/to/folder1/file1.grapto
  - /path/to/folder2/file1.grapto
```

**Next Steps**: Task 2 - File processing for ensembling (fetch source files, extract labels and bounding boxes)

---

## 2026-01-12: Parallel Processing for Pipeline

**Feature**: Allow collectra pipeline to run multiple files in parallel

**Changes Made**:
1. Added `--workers` / `-j` CLI option to `collectra run` command (default: 4 workers)
2. Added `process_file_in_subprocess()` function in `collectra/pipelines/base.py` - a top-level picklable function for ProcessPoolExecutor
3. Updated `_process_files()` method to use `ProcessPoolExecutor` for true parallelism (bypasses GIL)
4. Added `_get_optimal_workers()` method for resource-aware worker selection
5. Updated `_extract_run_options()` to include workers parameter

**Key Implementation Details**:
- Uses `ProcessPoolExecutor` with `spawn` context for complete process isolation
- Each file is processed in a separate subprocess with its own pipeline instance
- Falls back to sequential processing for single files or on resource exhaustion
- Graceful error handling - one file failure doesn't stop others

**Testing**:
- All 54 existing tests pass
- CLI option properly registered and accessible via `--workers` or `-j`
