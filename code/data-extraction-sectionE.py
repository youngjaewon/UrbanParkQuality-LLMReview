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
SECTION_C_CSV = "./fulltext_extraction_sectionC.csv"
OUTPUT_CSV = "./fulltext_extraction_sectionE_r35.csv"

PROMPT_TEXT = """
You are assisting a systematic literature review on equity in park and greenspace quality.

You will receive three things:
(1) A summary of how this study was previously coded in Section C (Quality Dimensions). This summary lists which aspects of park or greenspace quality were measured and analyzed in the Methods and Results sections.
(2) The full text of the article as a PDF.
(3) The response schema you must fill.

Your job is to code Section E (Equity). Use ONLY empirically analyzed content from the Methods and Results sections of the article. Do NOT infer, assume, or guess beyond what is explicitly analyzed. Do NOT rely on the Introduction or Discussion unless the paper clearly reports a measured and analyzed empirical result there. If there is no qualifying evidence, return an empty array [] or "NA" accordingly, and briefly explain in the detail fields why there is no qualifying evidence.

When filling any field whose name ends with "_Detail", you must briefly justify your coding using evidence from the Methods or Results. Whenever possible, include very short direct quotes (a phrase or short sentence) copied from the Methods or Results to show exactly how the article described it. If you cannot find a suitable quote in Methods or Results, you may closely paraphrase, but you must say that no direct quote was available. Do not invent quotes.

Key definitions and rules:

1. Target_User_Groups
Identify which specific social groups or populations the study analyzes in relation to park or greenspace quality. Examples include race or ethnicity, income groups, age groups, gender groups, health-related groups (for example disabled populations, people with obesity, neurodivergent people), housing status (for example renters, people in single-family housing, unhoused people), or users engaged in specific activities (for example mountain bikers).
If the study treats the public as a single undifferentiated whole and does not analyze any subpopulation, select "None (general public / unspecified users)".
Only include a category if the study actually measures or analyzes differences or outcomes for that group in Methods or Results. Do not include purely conceptual mentions in Introduction or Discussion.
If the study analyzes a group that does not fit any listed category, include "Other" and describe the group in Target_User_Groups_Detail.
In Target_User_Groups_Detail, explain which groups were analyzed and how, and include short direct quotes from Methods or Results where possible.

2. Distributive_Justice
Distributive justice refers to whether the quality of parks or greenspaces is unequally distributed across different population groups or across different geographic areas. For example, if the study finds that one neighborhood type or one demographic group receives systematically lower quality (for example fewer amenities, less shade, more safety issues), that is distributive justice.
Code which quality dimensions show unequal distribution. The dimensions are four high level categories from Section C:
- Ecological/Environmental
- Physical/Functional
- Social/Experiential
- Management/Governance
Only code a dimension if unequal distribution is directly analyzed in the Methods or Results, not just discussed conceptually.
In Distributive_Justice_Detail, for every dimension you mark, summarize the inequity and support it with short quotes from Methods or Results if available. Be concrete. For example, "lower amenity quality in lower income neighborhoods". If no distributive justice analysis was found, explain that and state that no direct quote exists.

3. Procedural_Justice
Procedural justice refers to who is involved in decision making, whose voices are included, and how participation in planning, design, management, or governance of parks is structured.
First, identify which quality dimensions (Ecological/Environmental, Physical/Functional, Social/Experiential, Management/Governance) are examined through a procedural justice lens. Only include a dimension if the study empirically analyzes participation, involvement, consultation, decision making, negotiation or planning practices related to that dimension in Methods or Results.
In Procedural_Justice_Detail, summarize how participation or decision making was analyzed for each selected dimension, and include short direct quotes from Methods or Results if possible.

Then fill Participation_Level. This should be an array of one or more of the following categories, based on Arnstein's ladder of participation:
- Informing
  One way information from officials to the public. No feedback channel. Symbolic notification only.
- Consultation
  The public is asked for opinions (for example survey, public hearing) but there is no guarantee those opinions will influence decisions.
- Placation
  Some community representatives are invited to advisory boards, but powerholders can easily overrule them.
- Partnership
  Citizens share power with officials through negotiation or joint structures. Real co decision.
- Delegated Power
  Citizens hold primary decision making authority over a specific program or plan.
- Citizen Control
  Citizens or community groups have full managerial control.

Participation_Level must reflect only what the study actually analyzes in Methods or Results.

Next, fill Participation_Outcome with one of:
- "Yes"   if the study evaluates whether participation produced tangible improvements in any aspect of park or greenspace quality (actual physical or managerial change, not just satisfaction with the process) and reports those improvements as empirical findings
- "No"    if participation is analyzed, but the study does not analyze any resulting change or improvement in park or greenspace quality
- "NA"    if no participation process was analyzed in the study at all

Then fill Participation_Detail. Participation_Detail must:
- justify Participation_Level and Participation_Outcome
- describe who participated and in what capacity
- explicitly state whether the article reports any resulting change in park quality
- include short direct quotes from Methods or Results whenever possible
If there was no analysis of participation at all, explain that explicitly and state that there is no direct quote.

If the study does not analyze any participation process in Methods or Results, then:
- Participation_Level should be []
- Participation_Outcome should be "NA"
- Participation_Detail should clearly state that no participation process was empirically analyzed and no quotes were available.

4. Recognitional_Justice
Recognitional justice concerns whether the study analyzes how park or greenspace quality affirms, respects, ignores, excludes, or misrepresents the identities, values, or cultural perspectives of particular groups.
Code which quality dimensions (Ecological/Environmental, Physical/Functional, Social/Experiential, Management/Governance) are discussed in terms of recognition or misrecognition. Recognition means features of the park reflect the group's cultural identity, needs, or values. Misrecognition means features ignore, exclude, or devalue those identities, needs, or values.
In Recognitional_Justice_Detail, for each dimension you mark, describe what was recognized or misrecognized and include short direct quotes from the Methods or Results if possible. If no recognitional justice analysis was found, explain that and state that no direct quote exists.

5. Use of Section C summary
You will be provided with a summary of how this same study was coded in Section C (Quality Dimensions), including which dimensions were measured and how. Use this summary as contextual guidance. Link your Section E coding to these same dimensions where appropriate.
However, you must still confirm that any claims about justice are empirically analyzed in this study's Methods or Results. If the Section C summary mentions a dimension but the article does not actually analyze justice for that dimension, then do not select that dimension for Distributive_Justice, Procedural_Justice, or Recognitional_Justice.

Return your answer in valid JSON following the provided response schema. Do not include any keys that are not in the schema. Do not include comments.
"""

# ----------------------------------------------------------
# 2. Response schema for Section E (Equity)
# ----------------------------------------------------------
schema = types.Schema(
    type="object",
    properties={

        # 22. Target User Groups
        "Target_User_Groups": types.Schema(
            type="array",
            items=types.Schema(
                type="string",
                enum=[
                    "None (general public / unspecified users)",
                    "Age",
                    "Race/ethnicity",
                    "Income",
                    "Gender",
                    "Health condition",
                    "Specific activity",
                    "Housing status",
                    "Other"
                ]
            ),
            description=(
                "Select all user or population groups that the study empirically analyzes in relation to park or "
                "greenspace quality in the Methods or Results sections. If the study does not analyze any specific "
                "group, use 'None (general public / unspecified users)'."
            )
        ),
        "Target_User_Groups_Detail": types.Schema(
            type="string",
            description=(
                "Briefly describe which groups were analyzed and how. Include short direct quotes from Methods or "
                "Results whenever possible. If 'Other' was selected, specify the group. If only general public was "
                "analyzed, explain that no specific groups were separately analyzed and note if no direct quote exists."
            )
        ),

        # 23. Distributive Justice
        "Distributive_Justice": types.Schema(
            type="array",
            items=types.Schema(
                type="string",
                enum=[
                    "Ecological/Environmental",
                    "Physical/Functional",
                    "Social/Experiential",
                    "Management/Governance"
                ]
            ),
            description=(
                "Which quality dimensions does the study empirically show to be unevenly distributed across "
                "population groups or places, based on Methods or Results. Select all that apply. "
                "Leave empty if no distributive justice analysis was performed."
            )
        ),
        "Distributive_Justice_Detail": types.Schema(
            type="string",
            description=(
                "For each dimension in Distributive_Justice, summarize the specific inequity that was identified in "
                "the Methods or Results. Include short direct quotes from Methods or Results whenever possible. "
                "If none were coded, explain that no distributive justice analysis was found and note if no direct quote exists."
            )
        ),

        # 24. Procedural Justice
        "Procedural_Justice": types.Schema(
            type="array",
            items=types.Schema(
                type="string",
                enum=[
                    "Ecological/Environmental",
                    "Physical/Functional",
                    "Social/Experiential",
                    "Management/Governance"
                ]
            ),
            description=(
                "Which quality dimensions does the study examine through a procedural justice lens, meaning stakeholder "
                "involvement, participation in decision making, planning input, negotiation, or governance processes. "
                "Select all that apply. Leave empty if there is no such analysis."
            )
        ),
        "Procedural_Justice_Detail": types.Schema(
            type="string",
            description=(
                "For each dimension in Procedural_Justice, describe how participation or decision making was analyzed "
                "in the Methods or Results. Include short direct quotes from Methods or Results whenever possible. "
                "If none were coded, explain that no procedural justice analysis was found and note if no direct quote exists."
            )
        ),

        # 24-2. Participation Level
        "Participation_Level": types.Schema(
            type="array",
            items=types.Schema(
                type="string",
                enum=[
                    "Informing",
                    "Consultation",
                    "Placation",
                    "Partnership",
                    "Delegated Power",
                    "Citizen Control"
                ]
            ),
            description=(
                "Select all levels of citizen or community participation analyzed in the Methods or Results. "
                "If no participatory process was analyzed at all, return an empty array []."
            )
        ),

        # 24-3. Participation Outcome
        "Participation_Outcome": types.Schema(
            type="string",
            enum=["Yes", "No", "NA"],
            description=(
                "If participation was analyzed, did the study evaluate whether participation produced tangible "
                "improvements in park or greenspace quality. 'Yes' if the study links participation to concrete "
                "quality improvements. 'No' if participation was analyzed but resulting quality improvements were "
                "not assessed. 'NA' if no participation process was analyzed."
            )
        ),

        # new: Participation Detail
        "Participation_Detail": types.Schema(
            type="string",
            description=(
                "Justify Participation_Level and Participation_Outcome using only Methods or Results. "
                "Describe who participated, how participation occurred, and whether the study reports any resulting "
                "change in park or greenspace quality. Include short direct quotes from Methods or Results whenever possible. "
                "If there was no participation process analyzed, clearly state that and note that no direct quote exists."
            )
        ),

        # 25. Recognitional Justice
        "Recognitional_Justice": types.Schema(
            type="array",
            items=types.Schema(
                type="string",
                enum=[
                    "Ecological/Environmental",
                    "Physical/Functional",
                    "Social/Experiential",
                    "Management/Governance"
                ]
            ),
            description=(
                "Which quality dimensions does the study analyze in terms of recognition or misrecognition of the "
                "cultural identities, needs, or values of specific groups. Select all that apply. "
                "Leave empty if there is no such analysis."
            )
        ),
        "Recognitional_Justice_Detail": types.Schema(
            type="string",
            description=(
                "For each dimension in Recognitional_Justice, summarize how recognition or misrecognition was described "
                "in the Methods or Results. Include short direct quotes from Methods or Results whenever possible. "
                "If none were coded, explain that no recognitional justice analysis was found and note that no direct quote exists."
            )
        ),
    }
)

# ----------------------------------------------------------
# helper 0. build Section C context for a single row
# ----------------------------------------------------------
def build_section_c_context(row):
    """
    Build a readable summary string from one df_c row.
    This is passed to the model as context.
    """
    def fmt_list(val):
        # val might already look like "['A', 'B']" which is fine as string
        if isinstance(val, str):
            return val
        if isinstance(val, list):
            return ", ".join(val)
        return str(val)

    parts = []
    parts.append("SECTION C SUMMARY FOR THIS STUDY")

    pqc = row.get("Park_Quality_Context", "")
    if isinstance(pqc, float) and pd.isna(pqc):
        pqc = ""
    parts.append(f"- Park_Quality_Context: {pqc}")

    eco_dim = row.get("Ecological_Environmental_Dimensions", "")
    eco_det = row.get("Ecological_Environmental_Dimensions_Detail", "")
    parts.append(f"- Ecological/Environmental Dimensions: {fmt_list(eco_dim)}")
    parts.append(f"  Detail: {eco_det}")

    phys_dim = row.get("Physical_Functional_Dimensions", "")
    phys_det = row.get("Physical_Functional_Dimensions_Detail", "")
    parts.append(f"- Physical/Functional Dimensions: {fmt_list(phys_dim)}")
    parts.append(f"  Detail: {phys_det}")

    soc_dim = row.get("Social_Experiential_Dimensions", "")
    soc_det = row.get("Social_Experiential_Dimensions_Detail", "")
    parts.append(f"- Social/Experiential Dimensions: {fmt_list(soc_dim)}")
    parts.append(f"  Detail: {soc_det}")

    mgmt_dim = row.get("Management_Governance_Dimensions", "")
    mgmt_det = row.get("Management_Governance_Dimensions_Detail", "")
    parts.append(f"- Management/Governance Dimensions: {fmt_list(mgmt_dim)}")
    parts.append(f"  Detail: {mgmt_det}")

    return "\n".join(parts)

# ----------------------------------------------------------
# helper 1. run model on a single pdf and return dict
# ----------------------------------------------------------
def process_single_pdf(client, pdf_path, df_c):
    """
    Upload one PDF, prepare Section C context, call the model with
    the global prompt and schema, parse the JSON response, and return
    (result_dict, elapsed_seconds).
    """
    if not os.path.exists(pdf_path):
        print(f"Skipping, file not found: {pdf_path}")
        return None, None

    start_time = time.time()
    file_basename = os.path.basename(pdf_path)

    # match Section C row
    row_match = df_c[df_c["File_Name"] == file_basename]
    if len(row_match) > 0:
        section_c_context = build_section_c_context(row_match.iloc[0])
    else:
        section_c_context = (
            "SECTION C SUMMARY FOR THIS STUDY\n"
            "- No Section C record was found for this file. "
            "If you cannot confirm justice related evidence from Methods or Results, "
            "return empty arrays, 'NA', and in all *_Detail fields explain that no qualifying "
            "evidence and no direct quotes were found."
        )

    # upload PDF
    try:
        uploaded_file = client.files.upload(file=pdf_path)
        print(f"Uploaded {file_basename}")

        time.sleep(0.5)

    except Exception as e:
        print(f"Upload failed for {pdf_path}: {e}")
        return None, None

    # call model
    try:
        response = client.models.generate_content(
            model="models/gemini-2.5-pro",
            contents=[
                types.Content(
                    parts=[
                        types.Part(text=PROMPT_TEXT),
                        types.Part(text=section_c_context),
                        types.Part(
                            file_data=types.FileData(
                                mime_type="application/pdf",
                                file_uri=uploaded_file.uri
                            )
                        ),
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

    # parse JSON
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        clean = re.sub(r"^json\s*", "", raw_text.strip(), flags=re.MULTILINE).strip()
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

    # cleanup remote file
    try:
        client.files.delete(name=uploaded_file.name)
    except Exception:
        pass

    # merge filename
    parsed = {
        "File_Name": file_basename,
        **parsed
    }

    elapsed = time.time() - start_time
    return parsed, elapsed

# ----------------------------------------------------------
# helper 2. loop over a folder of pdfs
# ----------------------------------------------------------
def process_folder(pdf_dir, output_csv, df_c):
    """
    Iterate over all PDFs in alphabetical order, skip ones already processed,
    call process_single_pdf, and save incremental checkpoints and final output.
    """
    client = genai.Client(api_key=API_KEY)
    rows = []
    per_file_times = {}

    # load existing out csv if present
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

    # collect pdf list
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
    #print(f"\nFound {total_files} PDF files in folder")
    print(f"\nManually specified {total_files} PDF files for processing")

    # loop
    for idx, fname in enumerate(pdf_files, start=1):
        if fname in processed_files:
            print(f"[{idx}/{total_files}] Skipping {fname} because it already exists in the CSV")
            continue

        fpath = os.path.join(pdf_dir, fname)
        print(f"\n[{idx}/{total_files}] Processing {fname}")

        result_dict, elapsed = process_single_pdf(client, fpath, df_c)

        if result_dict is not None:
            rows.append(result_dict)
            if elapsed is not None:
                print(f"Finished {fname} in {elapsed:.2f} seconds")
            else:
                print(f"Finished {fname}")
        else:
            rows.append({
                "File_Name": fname,
                "ERROR": "failed_to_extract"
            })
            if elapsed is not None:
                print(f"Failed {fname} in {elapsed:.2f} seconds")
            else:
                print(f"Failed {fname} with no timing captured")

        per_file_times[fname] = elapsed

        # checkpoint
        df_partial = pd.DataFrame(rows)
        df_partial.to_csv(output_csv, index=False)
        print(f"Checkpoint saved  current row count {len(df_partial)}")

    # final save
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
# main
# ----------------------------------------------------------
if __name__ == "__main__":
    df_c = pd.read_csv(SECTION_C_CSV)
    df_result = process_folder(PDF_DIR, OUTPUT_CSV, df_c)
    print(df_result)
