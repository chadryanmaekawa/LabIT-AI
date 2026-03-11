import os
import json
import re

import fitz # PyMuPDF
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# -----------------------------
# Setup
# -----------------------------
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="LabIT AI", layout="wide")
st.title("LabIT AI")
st.caption("LIS Build Copilot for Epic Beaker and Cerner PathNet")


# -----------------------------
# Helper functions
# -----------------------------
def get_openai_client():
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


def extract_pdf_text(uploaded_file) -> str:
    """Extract text from uploaded PDF."""
    pdf_bytes = uploaded_file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = []
    for page in doc:
        text.append(page.get_text())
    return "\n".join(text).strip()


def strip_code_fences(text: str) -> str:
    """Remove markdown code fences if model returns them."""
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def get_example_templates():
    return {
        "Custom / Blank": {
            "lis_target": "Epic Beaker",
            "test_name": "Respiratory Pathogen Panel",
            "instrument": "PCR multiplex platform",
            "methodology": "PCR/NAAT",
            "specimen": "Nasopharyngeal swab",
            "panel_size": 30,
            "notes": "Need discrete results per target, component table, validation checklist, and interface mapping.",
            "epic_fields": {
                "verification_required": "Tech verification",
                "mychart_release": "Yes",
            },
            "cerner_fields": {
            "result_status": "Final",
            "downtime_required": "No",
            },
        },
        "Respiratory Panel": {
            "lis_target": "Epic Beaker",
            "test_name": "Respiratory Pathogen Panel",
            "instrument": "Multiplex respiratory PCR platform",
            "methodology": "PCR/NAAT",
            "specimen": "Nasopharyngeal swab",
            "panel_size": 30,
            "notes": "Need discrete respiratory targets, validation checklist, and interface mapping.",
            "epic_fields": {
                "verification_required": "Tech verification",
                "mychart_release": "Yes",
            },
            "cerner_fields": {
                "result_status": "Final",
                "downtime_required": "No",
            },
        },
        "GI Panel": {
            "lis_target": "Cerner PathNet",
            "test_name": "GI Pathogen Panel",
            "instrument": "Multiplex GI PCR platform",
            "methodology": "PCR/NAAT",
            "specimen": "Stool",
            "panel_size": 30,
            "notes": "Need discrete GI targets, validation checklist, and interface mapping.",
            "epic_fields": {
                "verification_required": "Tech verification",
                "mychart_release": "No",
            },
            "cerner_fields": {
                "result_status": "Final",
                "downtime_required": "Yes",
            },
        },
        "STI Panel": {
            "lis_target": "Epic Beaker",
            "test_name": "STI Panel",
            "instrument": "Multiplex STI PCR platform",
            "methodology": "PCR/NAAT",
            "specimen": "Urogenital swab",
            "panel_size": 20,
            "notes": "Need discrete STI targets, result value consistency, and validation checklist.",
            "epic_fields": {
                "verification_required": "Tech verification",
                "mychart_release": "Delayed",
            },
            "cerner_fields": {
                "result_status": "Final",
                "downtime_required": "No",
            },
        },
        "Blood Culture ID Panel": {
            "lis_target": "Cerner PathNet",
            "test_name": "Blood Culture ID Panel",
            "instrument": "Rapid BCID molecular platform",
            "methodology": "PCR/NAAT",
            "specimen": "Positive blood culture bottle",
            "panel_size": 20,
            "notes": "Need organism targets, resistance markers, validation checklist, and interface mapping.",
            "epic_fields": {
            "verification_required": "Tech verification",
            "mychart_release": "No",
            },
            "cerner_fields": {
                "result_status": "Final",
                "downtime_required": "Yes",
            },
        },
    }


def default_mock_output(user_input: dict) -> dict:
    """Fallback output if AI JSON parsing fails."""
    return {
        "orderable_test_name": user_input.get("test_name", "RESP PANEL PCR"),
        "mnemonic_suggestions": ["RESPPCR", "RESPPANEL"],
        "panel_components": [
            {
                "name": "Influenza A",
                "result_values": ["Detected", "Not Detected"],
                "loinc": "",
                "interface_code": "FLUA",
            },
            {
                "name": "Influenza B",
                "result_values": ["Detected", "Not Detected"],
                "loinc": "",
                "interface_code": "FLUB",
            },
            {
                "name": "RSV",
                "result_values": ["Detected", "Not Detected"],
                "loinc": "",
                "interface_code": "RSV",
            },
        ],
        "validation_checklist": [
            "Verify orderable appears in workflow",
            "Confirm specimen routing",
            "Validate result component mapping",
            "Check interface message posting",
        ],
        "specimen_requirements": {
            "specimen_type": user_input.get("specimen", ""),
            "notes": "Review package insert for final specimen requirements.",
        },
        "interface_mapping": [
            {
                "component_name": "Influenza A",
                "instrument_code": "FLUA",
                "lis_component": "Influenza A",
            },
            {
                "component_name": "Influenza B",
                "instrument_code": "FLUB",
                "lis_component": "Influenza B",
            },
            {
                "component_name": "RSV",
                "instrument_code": "RSV",
                "lis_component": "RSV",
            },
        ],
        "interface_notes": [
            "Review OBX-3 and OBX-5 mappings in sample HL7 messages.",
            "Normalize result values before final LIS build.",
        ],
    }


def build_interface_code_lookup(interface_mapping: list) -> dict:
    lookup = {}
    for item in interface_mapping:
        comp_name = str(item.get("component_name", "")).strip().lower()
        instrument_code = str(item.get("instrument_code", "")).strip()
        if comp_name and instrument_code:
            lookup[comp_name] = instrument_code
        return lookup


def generate_lis_build(user_input: dict, package_insert_text: str, interface_doc_text: str) -> dict:
    """Call AI to generate structured LIS build output."""
    client = get_openai_client()
    if client is None:
        return default_mock_output(user_input)

    prompt = f"""
You are a laboratory informatics expert specializing in Epic Beaker and Cerner PathNet.

Generate an LIS test build draft from the user inputs, package insert text, and interface/analyzer documentation.

Return ONLY valid JSON with these keys:
- orderable_test_name
- mnemonic_suggestions
- panel_components
- validation_checklist
- specimen_requirements
- interface_mapping
- interface_notes

Rules:
- panel_components must be a list of objects
- each panel component object should include:
    - name
    - result_values
    - loinc
    - interface_code
- keep result_values simple, like ["Detected", "Not Detected"]
- if the package insert contains panel targets, extract them
- if interface documentation contains analyzer codes, map them
- interface_mapping must be a list of objects with:
    - component_name
    - instrument_code
    - lis_component
- interface_notes must be a list of concise implementation notes
- if information is missing, make reasonable assumptions
- do not include markdown
- do not include explanation outside JSON

User input:
{json.dumps(user_input, indent=2)}

Package insert text:
{package_insert_text[:15000]}

Interface / analyzer documentation text:
{interface_doc_text[:15000]}
""".strip()

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.2,
        messages=[
            {"role": "system", "content": "You are a laboratory informatics expert."},
            {"role": "user", "content": prompt},
        ],
    )

    raw = response.choices[0].message.content or ""
    raw = strip_code_fences(raw)

    try:
        parsed = json.loads(raw)
        return parsed
    except Exception:
        return default_mock_output(user_input)


def build_epic_ticket_markdown(parsed: dict, edited_df: pd.DataFrame, user_input: dict) -> str:
    lines = []
    lines.append(f"# Epic Beaker Build Ticket — {parsed.get('orderable_test_name', 'New Test Build')}")
    lines.append("")
    lines.append("## Build Summary")
    lines.append("- **LIS Target:** Epic Beaker")
    lines.append(f"- **Test Name:** {user_input.get('test_name', '')}")
    lines.append(f"- **Instrument / Platform:** {user_input.get('instrument', '')}")
    lines.append(f"- **Methodology:** {user_input.get('methodology', '')}")
    lines.append(f"- **Specimen Type:** {user_input.get('specimen', '')}")
    lines.append(f"- **Panel Size:** {user_input.get('panel_size', '')}")
    lines.append("")

    epic_fields = user_input.get("epic_fields", {})
    if epic_fields:
        lines.append("## Epic-Specific Settings")
        lines.append(f"- **Verification Required:** {epic_fields.get('verification_required', '')}")
        lines.append(f"- **MyChart Release:** {epic_fields.get('mychart_release', '')}")
        lines.append("")

    lines.append("## Mnemonic Suggestions")
    for item in parsed.get("mnemonic_suggestions", []):
        lines.append(f"- {item}")
    lines.append("")

    lines.append("## Component Build Table")
    if not edited_df.empty:
        lines.append("| Component Name | Result Values | LOINC | Interface Code |")
        lines.append("|---|---|---|---|")
        for _, row in edited_df.iterrows():
            lines.append(
                f"| {row.get('Component Name', '')} | {row.get('Result Values', '')} | "
                f"{row.get('LOINC', '')} | {row.get('Interface Code', '')} |"
            )
    else:
        lines.append("- No component rows available.")
    lines.append("")

    specimen_requirements = parsed.get("specimen_requirements", {})
    if specimen_requirements:
        lines.append("## Specimen Requirements")
        for key, value in specimen_requirements.items():
            lines.append(f"- **{key.replace('_', ' ').title()}:** {value}")
        lines.append("")

    if parsed.get("interface_notes"):
        lines.append("## Interface Notes")
        for note in parsed.get("interface_notes", []):
            lines.append(f"- {note}")
        lines.append("")

    lines.append("## Validation Checklist")
    for item in parsed.get("validation_checklist", []):
        lines.append(f"- [ ] {item}")

    return "\n".join(lines)


def build_cerner_ticket_markdown(parsed: dict, edited_df: pd.DataFrame, user_input: dict) -> str:
    lines = []
    lines.append(f"# Cerner PathNet Build Worksheet — {parsed.get('orderable_test_name', 'New Test Build')}")
    lines.append("")
    lines.append("## Build Summary")
    lines.append("- **LIS Target:** Cerner PathNet")
    lines.append(f"- **Test Name:** {user_input.get('test_name', '')}")
    lines.append(f"- **Instrument / Platform:** {user_input.get('instrument', '')}")
    lines.append(f"- **Methodology:** {user_input.get('methodology', '')}")
    lines.append(f"- **Specimen Type:** {user_input.get('specimen', '')}")
    lines.append(f"- **Panel Size:** {user_input.get('panel_size', '')}")
    lines.append("")

    cerner_fields = user_input.get("cerner_fields", {})
    if cerner_fields:
        lines.append("## Cerner-Specific Settings")
        lines.append(f"- **Default Result Status:** {cerner_fields.get('result_status', '')}")
        lines.append(f"- **Downtime Manual Entry Required:** {cerner_fields.get('downtime_required', '')}")
        lines.append("")

    lines.append("## Synonyms / Mnemonics")
    for item in parsed.get("mnemonic_suggestions", []):
        lines.append(f"- {item}")
    lines.append("")

    lines.append("## Component Worksheet")
    if not edited_df.empty:
        lines.append("| Component Name | Result Values | LOINC | Interface Code |")
        lines.append("|---|---|---|---|")
        for _, row in edited_df.iterrows():
            lines.append(
                f"| {row.get('Component Name', '')} | {row.get('Result Values', '')} | "
                f"{row.get('LOINC', '')} | {row.get('Interface Code', '')} |"
            )
    else:
        lines.append("- No component rows available.")
    lines.append("")

    if parsed.get("interface_mapping"):
        lines.append("## Interface Mapping")
        lines.append("| Component Name | Instrument Code | LIS Component |")
        lines.append("|---|---|---|")
        for item in parsed.get("interface_mapping", []):
            lines.append(
                f"| {item.get('component_name', '')} | {item.get('instrument_code', '')} | {item.get('lis_component', '')} |"
            )
        lines.append("")

    if parsed.get("interface_notes"):
        lines.append("## Interface Notes")
        for note in parsed.get("interface_notes", []):
            lines.append(f"- {note}")
        lines.append("")

    lines.append("## Validation Checklist")
    for item in parsed.get("validation_checklist", []):
        lines.append(f"- [ ] {item}")

    return "\n".join(lines)


def build_validation_plan_markdown(parsed: dict, user_input: dict) -> str:
    lines = []
    lines.append(f"# Validation Plan — {parsed.get('orderable_test_name', 'New Test Build')}")
    lines.append("")
    lines.append("## Test Overview")
    lines.append(f"- **Test Name:** {user_input.get('test_name', '')}")
    lines.append(f"- **LIS Target:** {user_input.get('lis_target', '')}")
    lines.append(f"- **Instrument / Platform:** {user_input.get('instrument', '')}")
    lines.append(f"- **Methodology:** {user_input.get('methodology', '')}")
    lines.append(f"- **Specimen Type:** {user_input.get('specimen', '')}")
    lines.append("")

    lines.append("## Validation Tasks")
    for item in parsed.get("validation_checklist", []):
        lines.append(f"- [ ] {item}")
    lines.append("")

    if parsed.get("interface_notes"):
        lines.append("## Interface Review Notes")
        for note in parsed.get("interface_notes", []):
            lines.append(f"- {note}")
        lines.append("")

    specimen_requirements = parsed.get("specimen_requirements", {})
    if specimen_requirements:
        lines.append("## Specimen Requirements")
        for key, value in specimen_requirements.items():
            lines.append(f"- **{key.replace('_', ' ').title()}:** {value}")

    return "\n".join(lines)


def apply_template(template_name: str):
    """Apply selected template and clear current build."""
    templates = get_example_templates()
    template_data = templates[template_name]

    st.session_state["selected_template"] = template_name
    st.session_state["form_lis_target"] = template_data["lis_target"]
    st.session_state["form_test_name"] = template_data["test_name"]
    st.session_state["form_instrument"] = template_data["instrument"]
    st.session_state["form_methodology"] = template_data["methodology"]
    st.session_state["form_specimen"] = template_data["specimen"]
    st.session_state["form_panel_size"] = template_data["panel_size"]
    st.session_state["form_notes"] = template_data["notes"]

    st.session_state["form_epic_verification_required"] = template_data.get("epic_fields", {}).get(
    "verification_required", "Tech verification"
    )
    st.session_state["form_epic_mychart_release"] = template_data.get("epic_fields", {}).get(
    "mychart_release", "Yes"
    )
    st.session_state["form_cerner_result_status"] = template_data.get("cerner_fields", {}).get(
    "result_status", "Final"
    )
    st.session_state["form_cerner_downtime_required"] = template_data.get("cerner_fields", {}).get(
    "downtime_required", "No"
    )

    st.session_state["component_table"] = pd.DataFrame(
    columns=["Component Name", "Result Values", "LOINC", "Interface Code"]
    )
    st.session_state["latest_output"] = None
    st.session_state["package_insert_text"] = ""
    st.session_state["interface_doc_text"] = ""
    st.session_state["last_user_input"] = {}
    st.session_state["reset_counter"] += 1


def reset_app_state():
    """Clear all current build state."""
    st.session_state["component_table"] = pd.DataFrame(
    columns=["Component Name", "Result Values", "LOINC", "Interface Code"]
    )
    st.session_state["latest_output"] = None
    st.session_state["package_insert_text"] = ""
    st.session_state["interface_doc_text"] = ""
    st.session_state["last_user_input"] = {}
    st.session_state["selected_template"] = "Custom / Blank"

    st.session_state["form_lis_target"] = "Epic Beaker"
    st.session_state["form_test_name"] = "Respiratory Pathogen Panel"
    st.session_state["form_instrument"] = "PCR multiplex platform"
    st.session_state["form_methodology"] = "PCR/NAAT"
    st.session_state["form_specimen"] = "Nasopharyngeal swab"
    st.session_state["form_panel_size"] = 15
    st.session_state["form_notes"] = (
    "Need discrete results per target, component table, validation checklist, and interface mapping."
    )
    st.session_state["form_epic_verification_required"] = "Tech verification"
    st.session_state["form_epic_mychart_release"] = "Yes"
    st.session_state["form_cerner_result_status"] = "Final"
    st.session_state["form_cerner_downtime_required"] = "No"
    st.session_state["reset_counter"] += 1


# -----------------------------
# Session state
# -----------------------------
if "component_table" not in st.session_state:
    st.session_state["component_table"] = pd.DataFrame(
        columns=["Component Name", "Result Values", "LOINC", "Interface Code"]
    )

if "latest_output" not in st.session_state:
    st.session_state["latest_output"] = None

if "package_insert_text" not in st.session_state:
    st.session_state["package_insert_text"] = ""

if "interface_doc_text" not in st.session_state:
    st.session_state["interface_doc_text"] = ""

if "last_user_input" not in st.session_state:
    st.session_state["last_user_input"] = {}

if "selected_template" not in st.session_state:
    st.session_state["selected_template"] = "Custom / Blank"
    
if "reset_counter" not in st.session_state:
    st.session_state["reset_counter"] = 0

if "show_reset_message" not in st.session_state:
    st.session_state["show_reset_message"] = False

if "form_lis_target" not in st.session_state:
    st.session_state["form_lis_target"] = "Epic Beaker"

if "form_test_name" not in st.session_state:
    st.session_state["form_test_name"] = "Respiratory Pathogen Panel"

if "form_instrument" not in st.session_state:
    st.session_state["form_instrument"] = "PCR multiplex platform"

if "form_methodology" not in st.session_state:
    st.session_state["form_methodology"] = "PCR/NAAT"

if "form_specimen" not in st.session_state:
    st.session_state["form_specimen"] = "Nasopharyngeal swab"

if "form_panel_size" not in st.session_state:
    st.session_state["form_panel_size"] = 15

if "form_notes" not in st.session_state:
    st.session_state["form_notes"] = (
        "Need discrete results per target, component table, validation checklist, and interface mapping."
    )

if "form_epic_verification_required" not in st.session_state:
    st.session_state["form_epic_verification_required"] = "Tech verification"

if "form_epic_mychart_release" not in st.session_state:
    st.session_state["form_epic_mychart_release"] = "Yes"

if "form_cerner_result_status" not in st.session_state:
    st.session_state["form_cerner_result_status"] = "Final"

if "form_cerner_downtime_required" not in st.session_state:
    st.session_state["form_cerner_downtime_required"] = "No"

if st.session_state.get("show_reset_message"):
    st.success("Ready for a new build.")
    st.session_state["show_reset_message"] = False
    

# -----------------------------
# Layout
# -----------------------------
left, right = st.columns([1, 1], gap="large")

with left:
    st.subheader("Example Templates")

    templates = get_example_templates()

    selected_template = st.selectbox(
        "Start from a template",
        list(templates.keys()),
        index=list(templates.keys()).index(st.session_state["selected_template"])
    )

    st.session_state["selected_template"] = selected_template
    template_data = templates[selected_template]

    with st.expander("Template Preview"):
        st.write(f"**Test Name:** {template_data['test_name']}")
        st.write(f"**LIS Target:** {template_data['lis_target']}")
        st.write(f"**Instrument:** {template_data['instrument']}")
        st.write(f"**Specimen:** {template_data['specimen']}")
        st.write(f"**Panel Size:** {template_data['panel_size']}")

    c1, c2 = st.columns(2)

    with c1:
        if st.button("Apply Template", type="secondary"):
            apply_template(selected_template)
            st.rerun()

    with c2:
        if st.button("Reset / New Build", type="secondary"):
            reset_app_state()
            st.session_state["show_reset_message"] = True
            st.rerun()

    st.subheader("Build Inputs")

    lis_target = st.selectbox(
        "LIS Target",
        ["Epic Beaker", "Cerner PathNet"],
        key="form_lis_target"
    )

    test_name = st.text_input(
        "Test Name",
        key="form_test_name"
    )

    instrument = st.text_input(
        "Instrument / Platform",
        key="form_instrument"
    )

    methodology = st.selectbox(
        "Methodology",
        ["PCR/NAAT", "Antigen", "Culture", "Serology"],
        key="form_methodology"
    )

    specimen = st.text_input(
        "Specimen Type",
        key="form_specimen"
    )

    panel_size = st.number_input(
        "Panel Size",
        min_value=1,
        max_value=50,
        key="form_panel_size"
    )

    st.subheader("LIS Specific Fields")

    epic_fields = {}
    cerner_fields = {}

    if lis_target == "Epic Beaker":
        epic_fields["verification_required"] = st.selectbox(
            "Verification Required",
            ["Tech verification", "Auto verify"],
            key="form_epic_verification_required"
        )

        epic_fields["mychart_release"] = st.selectbox(
            "MyChart Release",
            ["Yes", "No", "Delayed"],
            key="form_epic_mychart_release"
        )

    if lis_target == "Cerner PathNet":
        cerner_fields["result_status"] = st.selectbox(
            "Default Result Status",
            ["Final", "Preliminary"],
            key="form_cerner_result_status"
        )

        cerner_fields["downtime_required"] = st.selectbox(
            "Downtime Manual Entry Required?",
            ["Yes", "No"],
            key="form_cerner_downtime_required"
        )

    st.subheader("Upload Package Insert")
    package_insert_file = st.file_uploader(
        "Upload Package Insert PDF",
        type=["pdf"],
        key=f"package_insert_uploader_{st.session_state['reset_counter']}"
    )

    if package_insert_file is not None:
        try:
            st.session_state["package_insert_text"] = extract_pdf_text(package_insert_file)
            st.success("Package insert uploaded and processed.")
            with st.expander("Preview package insert text"):
                preview = st.session_state["package_insert_text"][:3000]
                st.text(preview if preview else "No text extracted.")
        except Exception as e:
            st.error(f"Could not read package insert PDF: {e}")
            st.session_state["package_insert_text"] = ""

    st.subheader("Upload Interface / Analyzer Documentation")
    interface_doc_file = st.file_uploader(
        "Upload Interface / Analyzer Spec PDF",
        type=["pdf"],
        key=f"interface_doc_uploader_{st.session_state['reset_counter']}"
    )

    if interface_doc_file is not None:
        try:
            st.session_state["interface_doc_text"] = extract_pdf_text(interface_doc_file)
            st.success("Interface / analyzer documentation uploaded and processed.")
            with st.expander("Preview interface document text"):
                preview = st.session_state["interface_doc_text"][:3000]
                st.text(preview if preview else "No text extracted.")
        except Exception as e:
            st.error(f"Could not read interface PDF: {e}")
            st.session_state["interface_doc_text"] = ""

with right:
    st.subheader("Generated Build")

    notes = st.text_area(
        "Extra Notes",
        key="form_notes"
    )

    generate = st.button("Generate Build Draft", type="primary")

    if generate:
        user_input = {
            "lis_target": lis_target,
            "test_name": test_name,
            "instrument": instrument,
            "methodology": methodology,
            "specimen": specimen,
            "panel_size": int(panel_size),
            "notes": notes,
            "epic_fields": epic_fields,
            "cerner_fields": cerner_fields
        }

        st.session_state["last_user_input"] = user_input

        if not api_key:
            st.error("OPENAI_API_KEY not found. Add it to your .env file.")
        else:
            if not st.session_state["package_insert_text"]:
                st.warning("No package insert uploaded. The build will be generated from form inputs only.")
            if not st.session_state["interface_doc_text"]:
                st.info("No interface document uploaded. Interface mapping will be inferred if possible.")

            with st.spinner("Generating LIS build draft..."):
                parsed = generate_lis_build(
                    user_input,
                    st.session_state["package_insert_text"],
                    st.session_state["interface_doc_text"]
                )
                st.session_state["latest_output"] = parsed

                interface_lookup = build_interface_code_lookup(parsed.get("interface_mapping", []))
                components = parsed.get("panel_components", [])
                table_rows = []

                for comp in components:
                    result_values = comp.get("result_values", ["Detected", "Not Detected"])
                    if isinstance(result_values, list):
                        result_values_str = " / ".join(result_values)
                    else:
                        result_values_str = str(result_values)

                    component_name = comp.get("name", "")
                    lookup_key = str(component_name).strip().lower()

                    interface_code = comp.get("interface_code", "")
                    if not interface_code:
                        interface_code = interface_lookup.get(lookup_key, "")

                    table_rows.append({
                        "Component Name": component_name,
                        "Result Values": result_values_str,
                        "LOINC": comp.get("loinc", ""),
                        "Interface Code": interface_code
                    })

                if table_rows:
                    st.session_state["component_table"] = pd.DataFrame(table_rows)
                else:
                    st.session_state["component_table"] = pd.DataFrame(
                        columns=["Component Name", "Result Values", "LOINC", "Interface Code"]
                    )

    if st.session_state["latest_output"] is not None:
        parsed = st.session_state["latest_output"]
        user_input_for_exports = st.session_state["last_user_input"]

        component_count = len(st.session_state["component_table"])
        package_insert_loaded = "Yes" if st.session_state["package_insert_text"] else "No"
        interface_doc_loaded = "Yes" if st.session_state["interface_doc_text"] else "No"
        lis_selected = user_input_for_exports.get("lis_target", "")
        orderable_name = parsed.get("orderable_test_name", "Not generated")

        st.subheader("Build Summary")
        s1, s2, s3, s4, s5 = st.columns(5)

        short_orderable = orderable_name[:20] + "..." if len(orderable_name) > 20 else orderable_name
        s1.metric("Orderable", short_orderable)
        s2.metric("Components", component_count)
        s3.metric("Package Insert", package_insert_loaded)
        s4.metric("Interface Doc", interface_doc_loaded)
        s5.metric("LIS Target", lis_selected)

        build_status = "✅ Ready" if parsed.get("orderable_test_name") else "❌ Missing"
        component_status = "✅ Ready" if component_count > 0 else "❌ Missing"
        interface_status = "✅ Ready" if parsed.get("interface_mapping") else "⚠️ Missing"
        export_status = "✅ Ready" if parsed.get("orderable_test_name") and component_count > 0 else "⚠️ Partial"

        st.subheader("Workflow Status")
        b1, b2, b3, b4 = st.columns(4)

        with b1:
            st.markdown(f"**Build Draft** \n{build_status}")
        with b2:
            st.markdown(f"**Components** \n{component_status}")
        with b3:
            st.markdown(f"**Interface Mapping** \n{interface_status}")
        with b4:
            st.markdown(f"**Export Package** \n{export_status}")

        st.divider()

        tab1, tab2, tab3, tab4 = st.tabs(
            ["Build Draft", "Component Builder", "Interface Mapping", "Exports"]
        )

        with tab1:
            st.subheader("AI Generated Build")
            st.json(parsed)

            st.subheader("Validation Checklist")
            for item in parsed.get("validation_checklist", []):
                st.write(f"- {item}")

            if "specimen_requirements" in parsed:
                st.subheader("Specimen Requirements")
                st.json(parsed["specimen_requirements"])

        with tab2:
            st.subheader("Component Builder")
            edited_df = st.data_editor(
                st.session_state["component_table"],
                num_rows="dynamic",
                use_container_width=True,
                key="component_editor"
            )
            st.session_state["component_table"] = edited_df

            st.subheader("Component Summary")
            st.dataframe(edited_df, use_container_width=True)

        with tab3:
            if parsed.get("interface_mapping"):
                st.subheader("Interface Mapping")
                interface_df = pd.DataFrame(parsed["interface_mapping"])
                st.dataframe(interface_df, use_container_width=True)
            else:
                st.info("No interface mapping available.")

            if parsed.get("interface_notes"):
                st.subheader("Interface Notes")
                for note in parsed["interface_notes"]:
                    st.write(f"- {note}")

        with tab4:
            edited_df = st.session_state["component_table"]

            json_data = json.dumps(parsed, indent=2).encode("utf-8")
            st.download_button(
                "Download Build JSON",
                data=json_data,
                file_name="lis_build_draft.json",
                mime="application/json"
            )

            csv_data = edited_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Component Table CSV",
                data=csv_data,
                file_name="component_builder_table.csv",
                mime="text/csv"
            )

            if parsed.get("interface_mapping"):
                interface_csv = pd.DataFrame(parsed["interface_mapping"]).to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download Interface Mapping CSV",
                    data=interface_csv,
                    file_name="interface_mapping.csv",
                    mime="text/csv"
                )

            epic_md = build_epic_ticket_markdown(parsed, edited_df, user_input_for_exports)
            cerner_md = build_cerner_ticket_markdown(parsed, edited_df, user_input_for_exports)
            validation_md = build_validation_plan_markdown(parsed, user_input_for_exports)

            st.subheader("Ticket Exports")
            st.download_button(
                "Download Epic Build Ticket (MD)",
                data=epic_md.encode("utf-8"),
                file_name="epic_build_ticket.md",
                mime="text/markdown"
            )

            st.download_button(
                "Download Cerner Worksheet (MD)",
                data=cerner_md.encode("utf-8"),
                file_name="cerner_build_worksheet.md",
                mime="text/markdown"
            )

            st.download_button(
                "Download Validation Plan (MD)",
                data=validation_md.encode("utf-8"),
                file_name="validation_plan.md",
                mime="text/markdown"
            )
