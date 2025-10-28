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
OUTPUT_CSV = "./fulltext_extraction_sectionD_r35.csv"

PROMPT_TEXT = """
You are assisting a systematic literature review.
Use only the attached PDF to fill the required metadata fields.
When describing outcomes, use wording that closely reflects how the article itself phrases and discusses them.
Quote or paraphrase expressions directly from the Methods or Results sections whenever possible,
rather than rephrasing them in your own words.

Code only outcomes that the article explicitly analyzes or reports in the Methods or Results sections
as *results or consequences* of park quality or greenspace quality — that is, variables or indicators that are measured
as dependent outcomes influenced by the level or characteristics of park or greenspace quality.
Do not code variables that represent components, dimensions, or indicators of park or greenspace quality itself.

Do not code outcomes for other related concepts such as general accessibility, usability, or satisfaction,
unless the article explicitly frames those concepts as dimensions of park or greenspace quality
and analyzes their outcomes as such.

Do not code anything mentioned only in the Introduction or Discussion without being empirically measured or analyzed.
Do not use any external information or assumptions.
If no qualifying information is found, clearly state that no relevant outcome information was identified and briefly explain why.
"""

# ----------------------------------------------------------
# 2. Response schema
# ----------------------------------------------------------
schema = types.Schema(
    type="object",
    properties={

        # 16. Regulating Ecosystem Services
        "Regulating_Ecosystem_Services": types.Schema(
            type="array",
            items=types.Schema(
                type="string",
                enum=[
                    "Climate regulation",
                    "Flood regulation/stormwater management",
                    "Disease regulation",
                    "Water purification",
                    "Air purification",
                    "Habitat provision",
                    "Other"
                ]
            ),
            description=(
                "Select all regulating ecosystem service (RES) outcomes that the study explicitly analyzes or measures "
                "as outcomes of park or greenspace quality in the Methods or Results sections. "
                "RES is defined as benefits obtained from the regulation of ecosystem processes, such as climate regulation, "
                "flood regulation or stormwater management, disease regulation, water purification, air purification, and habitat provision. "
                "Do not include anything mentioned only conceptually in the Introduction or Discussion."
            )
        ),
        "Regulating_Ecosystem_Services_Detail": types.Schema(
            type="string",
            description=(
                "Explain why each selected Regulating_Ecosystem_Services category qualifies as a measured or analyzed outcome of park or greenspace quality. "
                "Describe how it was directly analyzed in the Methods or Results sections. "
                "If 'Other' was selected, specify the regulating ecosystem service outcomes addressed and how they were analyzed."
            )
        ),

        # 17. Cultural Ecosystem Services
        "Cultural_Ecosystem_Services": types.Schema(
            type="array",
            items=types.Schema(
                type="string",
                enum=[
                    "Aesthetic",
                    "Spiritual",
                    "Educational",
                    "Recreational",
                    "Other"
                ]
            ),
            description=(
                "Select all cultural ecosystem service (CES) outcomes that the study explicitly analyzes or measures "
                "as outcomes of park or greenspace quality in the Methods or Results sections. "
                "CES is defined as nonmaterial benefits people obtain from ecosystems through spiritual enrichment, cognitive development, "
                "reflection, recreation, and aesthetic experiences. "
                "Aesthetic: beauty or aesthetic appreciation people find in various aspects of ecosystems, as reflected in support for parks, scenic drives, and housing location choices. "
                "Spiritual: spiritual and religious values assigned to ecosystems or their natural components, providing profound significance and meaning. "
                "Educational: components and processes that provide the basis for both formal and informal education, offering learning opportunities. "
                "Recreational: benefits people derive from ecosystems when choosing natural or cultivated landscapes for leisure activities and ecotourism. "
                "Do not include anything mentioned only conceptually in the Introduction or Discussion."
            )
        ),
        "Cultural_Ecosystem_Services_Detail": types.Schema(
            type="string",
            description=(
                "Explain why each selected Cultural_Ecosystem_Services category qualifies as a measured or analyzed outcome of park or greenspace quality. "
                "Describe how it was directly analyzed in the Methods or Results sections. "
                "If 'Other' was selected, specify the cultural ecosystem service outcomes addressed and how they were analyzed."
            )
        ),

        # 18. Provisioning Ecosystem Services
        "Provisioning_Ecosystem_Services": types.Schema(
            type="array",
            items=types.Schema(
                type="string",
                enum=[
                    "Food",
                    "Fresh water",
                    "Wood and fiber",
                    "Fuel",
                    "Other"
                ]
            ),
            description=(
                "Select all provisioning ecosystem service (PES) outcomes that the study explicitly analyzes or measures "
                "as outcomes of park or greenspace quality in the Methods or Results sections. "
                "PES is defined as products obtained from ecosystems, such as food, fresh water, wood and fiber, or fuel. "
                "Do not include anything mentioned only conceptually in the Introduction or Discussion."
            )
        ),
        "Provisioning_Ecosystem_Services_Detail": types.Schema(
            type="string",
            description=(
                "Explain why each selected Provisioning_Ecosystem_Services category qualifies as a measured or analyzed outcome of park or greenspace quality. "
                "Describe how it was directly analyzed in the Methods or Results sections. "
                "If 'Other' was selected, specify the provisioning ecosystem service outcomes addressed and how they were analyzed."
            )
        ),

        # 19. Health and Well-being
        "Health_and_Wellbeing": types.Schema(
            type="array",
            items=types.Schema(
                type="string",
                enum=[
                    "Physiological health",
                    "Psychological health"
                ]
            ),
            description=(
                "Select all health and well-being outcomes that the study explicitly analyzes or measures "
                "as outcomes of park or greenspace quality in the Methods or Results sections. "
                "Physiological health includes cardiovascular outcomes, obesity rates, biomarkers, and physical activity–related health effects. "
                "Psychological health includes affect, stress, cognition, and subjective well-being."
            )
        ),
        "Health_and_Wellbeing_Detail": types.Schema(
            type="string",
            description=(
                "Explain why each selected Health_and_Wellbeing category qualifies as a measured or analyzed outcome of park or greenspace quality. "
                "Describe how it was directly analyzed in the Methods or Results sections."
            )
        ),

        # 20. Social and Community Outcomes
        "Social_and_Community_Outcomes": types.Schema(
            type="array",
            items=types.Schema(
                type="string",
                enum=[
                    "Use, engagement, recreation",
                    "Social interaction, cohesion",
                    "Equity, environmental justice",
                    "Stewardship, education",
                    "Other"
                ]
            ),
            description=(
                "Select all social and community outcomes that the study explicitly analyzes or measures "
                "as outcomes of park or greenspace quality in the Methods or Results sections. "
                "Use, engagement, recreation refers to direct outcomes stemming from individuals’ own use of parks. "
                "Social interaction, cohesion, equity, environmental justice, and stewardship, education represent indirect outcomes "
                "emerging at the community or societal level and mediated through parks as spaces that facilitate broader social processes."
            )
        ),
        "Social_and_Community_Outcomes_Detail": types.Schema(
            type="string",
            description=(
                "Explain why each selected Social_and_Community_Outcomes category qualifies as a measured or analyzed outcome of park or greenspace quality. "
                "Describe how it was directly analyzed in the Methods or Results sections. "
                "If 'Other' was selected, specify the social and community outcomes addressed and how they were analyzed."
            )
        ),

        # 21. Economic Outcomes
        "Economic_Outcomes": types.Schema(
            type="array",
            items=types.Schema(
                type="string",
                enum=[
                    "Property value growth/capitalization",
                    "Economic impacts of recreation & tourism",
                    "Healthcare & societal cost savings",
                    "Economic benefits to individuals",
                    "Return on investment (ROI)",
                    "Other"
                ]
            ),
            description=(
                "Select all economic outcomes that the study explicitly analyzes or measures "
                "as outcomes of park or greenspace quality in the Methods or Results sections. "
                "Property value growth/capitalization refers to changes in nearby property values associated with parks. "
                "Economic impacts of recreation & tourism capture local or regional benefits generated by park visitors. "
                "Healthcare & societal cost savings include reductions in medical costs or public health expenditures. "
                "Economic benefits to individuals reflect improvements in financial wellbeing. "
                "Return on investment (ROI) refers to a comprehensive evaluation of total benefits relative to costs. "
                "Do not include anything mentioned only conceptually in the Introduction or Discussion."
            )
        ),
        "Economic_Outcomes_Detail": types.Schema(
            type="string",
            description=(
                "Explain why each selected Economic_Outcomes category qualifies as a measured or analyzed outcome of park or greenspace quality. "
                "Describe how it was directly analyzed in the Methods or Results sections. "
                "If 'Other' was selected, specify the economic outcomes addressed and how they were analyzed."
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
    and return (result_dict, elapsed_seconds).
    """
    if not os.path.exists(pdf_path):
        print(f"Skipping, file not found: {pdf_path}")
        return None, None

    start_time = time.time()

    # Upload the PDF
    try:
        uploaded_file = client.files.upload(file=pdf_path)
        print(f"Uploaded {os.path.basename(pdf_path)}")

        time.sleep(0.5)

    except Exception as e:
        print(f"Upload failed for {pdf_path}: {e}")
        return None, None

    # Call the model
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
        # Attempt cleanup
        try:
            client.files.delete(name=uploaded_file.name)
        except Exception:
            pass
        elapsed = time.time() - start_time
        return None, elapsed

    # Parse JSON from model response
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        # Try to clean common wrappers like a leading "json"
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

    # Always attempt file cleanup on the API side
    try:
        client.files.delete(name=uploaded_file.name)
    except Exception:
        pass

    # Attach file name to parsed result
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

    # Collect and sort PDF files alphabetically
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

        # Process a single PDF and measure elapsed time
        result_dict, elapsed = process_single_pdf(client, fpath)

        if result_dict is not None:
            rows.append(result_dict)
            print(f"Finished {fname} in {elapsed:.2f} seconds")
        else:
            rows.append({
                "File_Name": fname,
                "ERROR": "failed_to_extract"
            })
            if elapsed is not None:
                print(f"Failed {fname} in {elapsed:.2f} seconds")
            else:
                print(f"Failed {fname} with no timing captured")

        # Record per file timing
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
