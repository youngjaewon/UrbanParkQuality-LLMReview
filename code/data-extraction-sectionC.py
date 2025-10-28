# -*- coding: utf-8 -*-
import json
import re
import os
import time
import pandas as pd
from google import genai
from google.genai import types

# ----------------------------------------------------------
# 1. Configuration
# ----------------------------------------------------------
API_KEY = ""
PDF_DIR = "/Users/ywon3/ASU Dropbox/Youngjae Won/RESEARCH/Ongoing/GQEquityReview/paper-pdfs/for-review"
OUTPUT_CSV = "./fulltext_extraction_sectionC.csv"

PROMPT_TEXT = """
You are assisting a systematic literature review.
Use only the attached PDF to fill the required metadata fields.
Code only what the article itself explicitly treats as 'park quality' or 'greenspace quality.'
Only include quality dimensions that were directly measured or analyzed in the Methods or Results sections.
Do not code anything that was only mentioned conceptually in the Introduction or Discussion.
Do not code concepts that are not explicitly framed as 'quality' (for example 'usability,' 'accessibility,' or general satisfaction),
unless they are clearly defined and analyzed by the authors as dimensions of park or greenspace quality.
Do not use any external information or guesses.
If no relevant information can be found, clearly state that no qualifying content was identified and briefly explain why.
"""

# ----------------------------------------------------------
# 2. Response schema
# ----------------------------------------------------------
schema = types.Schema(
    type="object",
    properties={

        "Park_Quality_Definition": types.Schema(
            type="string",
            description=(
                "Describe how the study defined 'park quality' or 'greenspace quality' in its own terms. "
                "Do not infer or generalize beyond what is explicitly stated in the article."
            )
        ),

        "Ecological_Environmental_Dimensions": types.Schema(
            type="array",
            items=types.Schema(
                type="string",
                enum=[
                    "Biodiversity and/or habitat",
                    "Vegetation and/or flora",
                    "Soil quality",
                    "Water quality and/or hydrology",
                    "Air quality and/or microclimate",
                    "Acoustic environment & soundscape",
                    "Other"
                ]
            ),
            description=(
                "Select all ecological or environmental aspects that the study explicitly treats as part of "
                "'park quality' or 'greenspace quality' and that were directly measured or analyzed in the "
                "Methods or Results sections. "
                "Do not include anything mentioned only in the Introduction or Discussion without being measured."
            )
        ),
        "Ecological_Environmental_Dimensions_Detail": types.Schema(
            type="string",
            description=(
                "Explain why each selected Ecological_Environmental_Dimensions category qualifies as park or greenspace 'quality' in this study. "
                "Describe how it was directly measured or analyzed in the Methods or Results sections. "
                "When describing how each aspect was defined or measured, use direct quotes from the article whenever possible, "
                "especially wording that appears in the Methods or Results. "
                "If 'Other' was selected, name the ecological or environmental aspect and describe how it was measured."
            )
        ),

        "Physical_Functional_Dimensions": types.Schema(
            type="array",
            items=types.Schema(
                type="string",
                enum=[
                    "Facilities and/or amenities",
                    "Maintenance and/or cleanliness",
                    "Safety and/or security",
                    "Design and/or aesthetics",
                    "Size and/or acreage",
                    "Internal accessibility",
                    "External accessibility",
                    "Other"
                ]
            ),
            description=(
                "Select all physical or functional aspects that the study explicitly treats as part of "
                "'park quality' or 'greenspace quality' and that were directly measured or analyzed in the "
                "Methods or Results sections. "
                "Only include features that were operationalized with data in this article. "
                "'Internal accessibility' refers to ease of movement within the park such as ADA or disability access, "
                "pathways, signage, or internal connectivity. "
                "'External accessibility' refers to how easily users can reach or recognize the park from outside "
                "such as proximity to transit stops, pedestrian or cycling routes, parking availability, "
                "or visibility of entrances. "
                "Do not include anything mentioned only in the Introduction or Discussion without being measured."
            )
        ),
        "Physical_Functional_Dimensions_Detail": types.Schema(
            type="string",
            description=(
                "Explain why each selected Physical_Functional_Dimensions category qualifies as park or greenspace 'quality' in this study. "
                "Describe how it was directly measured or analyzed in the Methods or Results sections. "
                "When describing how each aspect was defined or measured, use direct quotes from the article whenever possible, "
                "especially wording that appears in the Methods or Results. "
                "If 'Other' was selected, name the physical or functional aspect and describe how it was measured."
            )
        ),

        "Social_Experiential_Dimensions": types.Schema(
            type="array",
            items=types.Schema(
                type="string",
                enum=[
                    "Perceived quality",
                    "Recreation and/or leisure opportunities or programming",
                    "Social interactions and/or community-building",
                    "Cultural and/or educational features",
                    "Stewardship behavior",
                    "Other"
                ]
            ),
            description=(
                "Select all social or experiential aspects that the study explicitly treats as part of "
                "'park quality' or 'greenspace quality' and that were directly measured or analyzed in the "
                "Methods or Results sections. "
                "'Stewardship behavior' includes volunteering or personal commitment to park maintenance and care. "
                "Do not include anything mentioned only in the Introduction or Discussion without being measured."
            )
        ),
        "Social_Experiential_Dimensions_Detail": types.Schema(
            type="string",
            description=(
                "Explain why each selected Social_Experiential_Dimensions category qualifies as park or greenspace 'quality' in this study. "
                "Describe how it was directly measured or analyzed in the Methods or Results sections. "
                "When describing how each aspect was defined or measured, use direct quotes from the article whenever possible, "
                "especially wording that appears in the Methods or Results. "
                "If 'Other' was selected, name the social or experiential aspect and describe how it was measured."
            )
        ),

        "Management_Governance_Dimensions": types.Schema(
            type="array",
            items=types.Schema(
                type="string",
                enum=[
                    "Planning and/or policy",
                    "Citizen participation and/or collaboration",
                    "Funding and/or resource allocation",
                    "Other"
                ]
            ),
            description=(
                "Select all management or governance aspects that the study explicitly treats as part of "
                "'park quality' or 'greenspace quality' and that were directly measured or analyzed in the "
                "Methods or Results sections. "
                "Management or governance quality refers to institutional capacity or operational performance "
                "that ensures the park provides its intended benefits. "
                "Only include items that were operationalized with data in this article. "
                "Do not include anything mentioned only in the Introduction or Discussion without being measured."
            )
        ),
        "Management_Governance_Dimensions_Detail": types.Schema(
            type="string",
            description=(
                "Explain why each selected Management_Governance_Dimensions category qualifies as park or greenspace 'quality' in this study. "
                "Describe how it was directly measured or analyzed in the Methods or Results sections. "
                "When describing how each aspect was defined or measured, use direct quotes from the article whenever possible, "
                "especially wording that appears in the Methods or Results. "
                "If 'Other' was selected, name the management or governance aspect and describe how it was measured."
            )
        ),
    }
)

# ----------------------------------------------------------
# helper 1. run model on a single pdf and return dict
# ----------------------------------------------------------
def process_single_pdf(client, pdf_path):
    """
    Upload one PDF, call the model with the global prompt and schema,
    parse the JSON response, clean up the uploaded file on the API side,
    and return a dictionary with extracted fields plus file name.
    """
    if not os.path.exists(pdf_path):
        print(f"Skipping, file not found: {pdf_path}")
        return None, None

    start_time = time.time()

    try:
        uploaded_file = client.files.upload(file=pdf_path)
        print(f"Uploaded {os.path.basename(pdf_path)}")

        time.sleep(0.5)

    except Exception as e:
        print(f"Upload failed for {pdf_path}: {e}")
        return None, None

    try:
        response = client.models.generate_content(
            model="models/gemini-2.5-pro",
            contents=[
                types.Content(
                    parts=[
                        types.Part(text=PROMPT_TEXT),
                        types.Part(
                            file_data=types.FileData(
                                mime_type="application/pdf",
                                file_uri=uploaded_file.uri
                            )
                        )
                    ]
                )
            ],
            config=types.GenerateContentConfig(
                temperature=0,
                responseSchema=schema,
                response_mime_type="application/json",
            ),
        )
        raw_text = response.text
    except Exception as e:
        print(f"Model call failed for {pdf_path}: {e}")
        try:
            client.files.delete(name=uploaded_file.name)
        except Exception:
            pass
        elapsed = time.time() - start_time
        return None, elapsed

    # Try to parse JSON
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        # attempt to clean common wrappers such as leading "json"
        clean = re.sub(r"^json\s*|$", "", raw_text.strip(), flags=re.MULTILINE).strip()
        start_idx, end_idx = clean.find("{"), clean.rfind("}")
        if start_idx != -1 and end_idx != -1:
            try:
                parsed = json.loads(clean[start_idx:end_idx + 1])
            except json.JSONDecodeError:
                print(f"JSON parse failed for {pdf_path}")
                try:
                    client.files.delete(name=uploaded_file.name)
                except Exception:
                    pass
                elapsed = time.time() - start_time
                return None, elapsed
        else:
            print(f"No valid JSON object in model response for {pdf_path}")
            try:
                client.files.delete(name=uploaded_file.name)
            except Exception:
                pass
            elapsed = time.time() - start_time
            return None, elapsed

    # Always attempt cleanup
    try:
        client.files.delete(name=uploaded_file.name)
    except Exception:
        pass

    parsed = {
        "File_Name": os.path.basename(pdf_path),
        **parsed
    }

    elapsed = time.time() - start_time
    return parsed, elapsed

# ----------------------------------------------------------
# helper 2. loop over a folder of pdfs
# ----------------------------------------------------------
def process_folder(pdf_dir, output_csv):
    """
    Create a client, iterate through all PDFs in the directory in
    alphabetical order, skip files that were already processed in the
    existing CSV if present, time each file, and save checkpoints
    after each file. Return the final DataFrame.
    """
    client = genai.Client(api_key=API_KEY)
    rows = []
    per_file_times = {}

    # Load existing CSV if present
    processed_files = set()
    if os.path.exists(output_csv):
        try:
            prev_df = pd.read_csv(output_csv)
            rows = prev_df.to_dict(orient="records")
            if "File_Name" in prev_df.columns:
                processed_files = set(prev_df["File_Name"].astype(str).tolist())
            print(f"Loaded {len(rows)} existing rows from {output_csv}")
        except Exception as e:
            print(f"Could not load existing CSV: {e}")
            rows = []
            processed_files = set()

    # # Collect and sort PDF files alphabetically
    # pdf_files = sorted(
    #     [f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")],
    #     key=str.lower
    # )

    pdf_files = [
        'uebel2025.pdf',
        'coisnon2024.pdf',
        'bajwoluk2023.pdf',
        'huzlik2020.pdf',
        'hadavi2018.pdf',
        'gatti2022.pdf',
        'dandolo2022.pdf',
        'ziemelniece2023.pdf',
        'cheng2020.pdf',
        'ward2023.pdf',
        'roe2016.pdf',
        'cengiz2012.pdf',
        'southon2017.pdf',
        'mceachan2018.pdf',
        'battisti2020a.pdf',
        'banda2014.pdf',
        'chen2019b.pdf',
        'fornal-pieniak2023.pdf',
        'ghanem2024.pdf',
        'putra2021b.pdf',
        'stanley2022.pdf',
        'baka2022.pdf',
        'sander2017.pdf',
        'feng2017a.pdf',
        'yang2024b.pdf',
        'vandillen2012.pdf',
        'wu2025b.pdf',
        'mccann2021.pdf',
        'fors2015.pdf',
        'mullenbach2022.pdf',
        'song2020.pdf',
        'irvine2013.pdf',
        'arnberger2012.pdf',
        'dobbinson2020.pdf',
        'wood2018.pdf'
    ]


    total_files = len(pdf_files)
    print(f"\nManually specified {total_files} PDF files for processing")

    #print(f"\nFound {total_files} PDF files in folder")

    # Iterate through the PDFs
    for idx, fname in enumerate(pdf_files, start=1):
        if fname in processed_files:
            print(f"[{idx}/{total_files}] Skipping {fname} because it already exists in the CSV")
            continue

        fpath = os.path.join(pdf_dir, fname)
        print(f"\n[{idx}/{total_files}] Processing {fname} ...")

        result_dict, elapsed = process_single_pdf(client, fpath)

        if result_dict is not None:
            rows.append(result_dict)
            print(f"Finished {fname} in {elapsed:.2f} seconds")
        else:
            rows.append({
                "File_Name": fname,
                "ERROR": "failed_to_extract"
            })
            # even if extraction failed we still record the timing if we have it
            if elapsed is not None:
                print(f"Failed {fname} in {elapsed:.2f} seconds")
            else:
                print(f"Failed {fname} with no timing captured")
        per_file_times[fname] = elapsed

        # Save checkpoint after each file
        df_partial = pd.DataFrame(rows)
        df_partial.to_csv(output_csv, index=False)
        print(f"Checkpoint saved  current row count {len(df_partial)}")

    # Final save
    df_all = pd.DataFrame(rows)
    df_all.to_csv(output_csv, index=False)

    print("\nProcessing summary")
    print(f"Total files in folder  {total_files}")
    print(f"Total rows written     {len(df_all)}")
    print(f"Saved final output to  {output_csv}")

    print("\nPer file elapsed time in seconds")
    for fname, tval in per_file_times.items():
        if tval is None:
            print(f"{fname}: no timing recorded")
        else:
            print(f"{fname}: {tval:.2f} sec")

    return df_all

# ----------------------------------------------------------
# main entry
# ----------------------------------------------------------
if __name__ == "__main__":
    df_result = process_folder(PDF_DIR, OUTPUT_CSV)
    print(df_result)
