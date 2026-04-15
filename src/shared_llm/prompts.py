"""Prompt templates for LLM-based extraction.

OCR path: LLM receives TOON-encoded OCR text and returns {v, wids} per field.
FILE_ONLY path: LLM receives the raw file and returns {v, bbox, page} per field.
"""

OCR_PROMPT_TEMPLATE = """
    \n\n
    ### OCR Structure Explanation ###

    **Words:**
    - Each entry in `words[...]{{id,text,x1,y1,x2,y2}}` represents one OCR word.
    - (`x1`, `y1`, `x2`, `y2`) are the bounding box coordinates normalized to a 0-1 scale.
    - `text` is the word text.
    - `id` is the unique word identifier used for referencing.

    **Tables:**
    - Each entry in `tables[...]{{rows,cols,cells,box}}` represents a detected table structure.
    - `rows` and `cols` indicate the table dimensions.
    - `cells` is a list of table cells, each containing:
      - `row`, `col`: cell position (0-indexed)
      - `row_span`, `col_span`: cell span (default 1)
      - `text`: cell content
      - `kind`: cell type (e.g., header, data)
      - `box`: cell bounding box coordinates (normalized 0-1 scale) if available
    - `box`: overall table bounding box (normalized 0-1 scale) if available.

    **OCR DATA:**
    {ocr_toon}
    \n\n
"""

USER_INSTRUCTIONS_TEMPLATE = """
    \n\n
    ### MANDATORY INSTRUCTIONS ###
    You are an automated data processor. Your response MUST strictly adhere to the following rules without deviation.
    1.  Execute the User's Task: Process the request detailed below in the "User's Task" section.
    2.  Use User-Defined Field Names: If the user provides a specific name for an attribute field or a schema containing field names, you MUST use those exact names as the attribute names in your output.
    User's Task (Follow Exactly):
    {user_instructions}
    \n\n
"""

FIXED_SYSTEM_PROMPT = """
    \n\n
    ###SOURCE OF TRUTH AND DATA INTEGRITY ###

    *   **THE OCR IS THE ONLY SOURCE OF TRUTH**: Every value you output for `v` and `wids` MUST be directly derived from the provided OCR. Do not infer, guess, or hallucinate any data.
    *   **STRICT ADHERENCE TO OCR DATA**: You MUST use the `text` and `word_id` values exactly as they appear in the OCR's `w` (words) list.

    ---

    ### 2. EXTRACTION AND CALCULATION RULES ###

    For every required field, you will perform the following steps to populate its corresponding `{"v": ..., "wids": ...}` object.

    **Step 1: Locate the Information**
    Find the relevant word or group of words for the target field within the `w` (words) list of the OCR data.

    **Step 2: Populate Output Values**

    #### **RULES FOR POPULATING THE OUTPUT OBJECT**

    1.  **`v` (value: String)**:
        *   Combine the `text` from one or more OCR word objects to form the complete string value. The words should be joined by a single space.

    2.  **`wids` (word_ids: List[Integer])**:
        *   Collect the `word_id` from each constituent OCR word object.
        *   The list of `word_id`s MUST be in the same order as the words appear in the final `v` string.
    \n\n

    Output **ONLY** a valid JSON object as per the specified schema.
    Do not include any additional text, explanations, or formatting outside the JSON structure.
    Use "Not found" for missing data.
"""

FIXED_SYSTEM_PROMPT_FILE_ONLY = """
    \n\n
    ### SOURCE OF TRUTH AND DATA INTEGRITY ###

    *   **THE PROVIDED FILE IS THE ONLY SOURCE OF TRUTH**: Every value you output MUST be directly derived from the content visible in the provided file. Do not infer, guess, or hallucinate any data.

    ---

    ### 2. EXTRACTION AND BOUNDING BOX RULES ###

    For every required field, you will perform the following steps to populate its corresponding `{"v": ..., "bbox": ..., "page": ...}` object.

    #### **RULES FOR POPULATING THE OUTPUT OBJECT**

    1.  **`v` (value: String)**: The extracted text value for the field.

    2.  **`bbox` (bounding_box: Object | null)**:
        *   Provide the normalized bounding box with keys: `x1`, `y1`, `x2`, `y2` (0-1 scale).
        *   If the bounding box cannot be determined, use `null`.

    3.  **`page` (page_number: Integer)**: The 1-based page number where the value was found.
    \n\n

    Output **ONLY** a valid JSON object as per the specified schema.
    Do not include any additional text, explanations, or formatting outside the JSON structure.
    Use "Not found" for missing data.
"""
