import os
import json
import datetime

def generate_qc_report(config_path='config.json', report_path=None):
    """Generate a simple HTML QC report summarising the pipeline inputs and checks.
    The function reads the config, records timestamps, and writes a readable report.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f'Config not found: {config_path}')
    with open(config_path) as f:
        config = json.load(f)
    if report_path is None:
        report_path = os.path.join(config['paths']['output_dir'], 'qc_report.html')
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    html = f'''<html><head><title>QC Report</title></head><body>
    <h1>Risk-Map Pipeline QC Report</h1>
    <p><strong>Generated:</strong> {now}</p>
    <h2>Patient / Data Paths</h2>
    <ul>
        <li>Patient ID: {config.get('patient_id','N/A')}</li>
        <li>MRI: {config['paths'].get('mri_image','N/A')}</li>
        <li>Tumor mask: {config['paths'].get('tumor_mask','N/A')}</li>
        <li>SynthSeg labels: {config['paths'].get('synthseg_labels','N/A')}</li>
    </ul>
    <h2>Outputs</h2>
    <ul>
        <li>Risk map: {config['paths'].get('risk_map_nii','N/A')}</li>
        <li>Structure table: {config['paths'].get('structure_table_csv','N/A')}</li>
        <li>Visualization PNG: {config['paths'].get('visualization_png','N/A')}</li>
    </ul>
    </body></html>'''
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'[QC] Report written to {report_path}')
    return report_path
