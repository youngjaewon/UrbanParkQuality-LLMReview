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
OUTPUT_CSV = "./fulltext_extraction_AB_r35.csv"

PROMPT_TEXT = """
You are assisting a systematic literature review.
Use only the attached PDF to fill the required metadata fields.
For explanations, use direct quotes from the article whenever possible, rather than rephrasing them in your own words.
Do not use any external information or guesses.
"""

# ----------------------------------------------------------
# 2. Response schema
# ----------------------------------------------------------
schema = types.Schema(
    type="object",
    properties={
        "Title": types.Schema(
            type="string",
            description="Title of the study."
        ),
        "Lead_Author": types.Schema(
            type="string",
            description=(
                "Name of the first (lead) author of the study, formatted as 'Last Name, First Name'. "
                "If a middle name or initial is present, include it between the first and last name, e.g., 'Smith, John A.'."
            )
        ),
        "Year": types.Schema(
            type="integer",
            description="Year that the study was published."
        ),
        "Journal": types.Schema(
            type="string",
            description="Name of the journal where the study was published."
        ),
        "Country": types.Schema(
            type="string",
            description=(
                "Country or countries where parks or greenspaces are located. "
                "Use standardized full country names in English (e.g., 'United States', 'France'). "
                "If multiple countries are listed, separate them with semicolons. "
                "If not explicitly reported, write 'Not reported'."
            )
        ),
        "Country_ISO3": types.Schema(
            type="string",
            description=(
                "Provide the corresponding ISO 3166-1 alpha-3 code(s) for the country names above. "
                "For example, 'United States' → 'USA', 'South Korea' → 'KOR', 'France' → 'FRA'. "
                "If multiple countries are listed, separate codes with semicolons in the same order. "
                "If the country is 'Not reported', write 'Not reported'."
            )
        ),
        "City": types.Schema(
            type="string",
            description="City or cities where parks are located. If there are multiple cities, separate them with semicolons."
        ),
        "Spatial_Scale_of_Analysis": types.Schema(
            type="string",
            enum=[
                "Single greenspace/park",
                "Multiple greenspaces/parks",
                "Entire network or system"
            ],
            description=(
                "Spatial scope at which park or greenspace quality is examined. Choose one of the following categories:"
                "Single greenspace/park"
                "Multiple greenspaces/parks: two or more parks within a defined area such as a neighborhood or city"
                "Entire network or system: all greenspaces/parks across a metropolitan area"
            )
        ),
        "Number_of_Parks": types.Schema(
            any_of=[
                types.Schema(type="integer"),
                types.Schema(type="string", enum=["Not reported"])
            ],
            description=(
                "How many distinct park or greenspace units were included in the study analysis. "
                "Enter a numeric value. Use 'Not reported' if the number of parks is not specified in the study."
            )
        ),
        "Spatial_Scale_Detail": types.Schema(
            type="string",
            description="Provide justification for why the Spatial_Scale_of_Analysis and Number_of_Parks categories were chosen."
        ),
        "Article_Type": types.Schema(
            type="string",
            enum=[
                "Empirical",
                "Methodological",
                "Literature Review",
                "Theoretical",
                "Other"
            ],
            description=(
                "Categorize each study based on its primary objective or research question."
                "Choose ONLY ONE option that best describes the study:"
                "Empirical: Studies that apply established tools or methods to evaluate specific parks, greenspaces, or surrounding environments."
                "Methodological: Studies that develop new instruments, metrics, or indices."
                "Literature Review: Studies that synthesize existing research."
                "Theoretical: Studies that develop new theories or conceptual frameworks."
            ),
        ),
        "Article_Type_Detail": types.Schema(
            type="string",
            description=(
                "Provide justification for why the Article_Type category was chosen. "
                "If Other was selected for Article_Type, clearly define the alternative category and describe it in detail."
            )
        ),
        "Data_Collection_Method": types.Schema(
            type="array",
            items=types.Schema(
                type="string",
                enum=[
                    "On-site audit or observation",
                    "Questionnaire surveys",
                    "Interviews and/or focus groups",
                    "Experiments",
                    "Geospatial data and/or remote sensing",
                    "Ecological measurement",
                    "Citizen science and/or user-generated data",
                    "Literature and document review",
                    "Research through design (often implemented in design studios)",
                    "Other"
                ]
            ),
            description=(
                "The data collection methods used in the study. Select all options that apply."
                "Note: 'Citizen science and/or user-generated data' refers to data passively collected from users (e.g., social media posts, online reviews) or data actively collected by the public as part of a formal citizen science project. This category does not include qualitative data actively solicited by researchers, such as written reflections in response to a prompt, which should be categorized as 'Questionnaire surveys' or 'Interviews and/or focus groups'."
            )
        ),
        "Data_Collection_Method_Detail": types.Schema(
            type="string",
            description=(
                "Provide justification for why the Data_Collection_Method categories were chosen. "
                "If Other was selected for Data_Collection_Method, clearly define the alternative data collection method and describe it in detail."
            )
        ),
        "Data_Analysis_Method": types.Schema(
            type="array",
            items=types.Schema(
                type="string",
                enum=[
                    "Quantitative Statistical Modeling",
                    "Qualitative Content Analysis",
                    "Spatial Analysis",
                    "Causal Inference & Experimental Design",
                    "Computational & AI-based Analysis",
                    "Other"
                ]
            ),
            description=(
                "Categorize each study based on its data analysis method(s). Select all options that apply:"
                "Quantitative Statistical Modeling: Focuses on quantifying statistical relationships between variables (for example correlation, regression) or testing for significant differences between groups. The primary goal is to explore and explain associations within the data."
                "Qualitative Content Analysis: Interprets textual data such as interviews or focus groups to identify and analyze underlying themes, patterns, and meanings. The goal is to understand context, perceptions, and narratives in the qualitative data."
                "Spatial Analysis: Uses geographic location data as a core component to statistically analyze spatial patterns. This includes examining geographic distribution, density, accessibility, clustering, and spatial autocorrelation. Simply mapping data does not fall into this category."
                "Causal Inference & Experimental Design: Aims to infer the causal effect of an intervention or change in conditions on an outcome. The study is structured to evaluate this effect, for example, through experimental or quasi-experimental designs (e.g., pre-post comparisons), often by establishing treatment and control or comparison groups."
                "Computational & AI-based Analysis: Uses computational algorithms, such as machine learning and natural language processing, to learn from and identify patterns within large scale, often unstructured, data such as text or images. This approach aims to predict outcomes or automatically classify complex information."
            ),
        ),
        "Data_Analysis_Method_Detail": types.Schema(
            type="string",
            description=(
                "Provide justification for why the Data_Analysis_Method categories were chosen. "
                "If Other was selected for Data_Analysis_Method, clearly define the alternative category and describe it in detail."
            )
        ),
    },
    required=[
        "Title",
        "Lead_Author",
        "Year",
        "Journal",
        "Country",
        "Country_ISO3",
        "City",
        "Spatial_Scale_of_Analysis",
        "Number_of_Parks",
        "Spatial_Scale_Detail",
        "Article_Type",
        "Article_Type_Detail",
        "Data_Collection_Method",
        "Data_Collection_Method_Detail",
        "Data_Analysis_Method",
        "Data_Analysis_Method_Detail"
    ]
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
