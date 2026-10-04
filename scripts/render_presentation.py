"""
Render PowerPoint presentation to high-resolution PNG slides using PowerPoint COM.
"""
import os
import win32com.client

def render_pptx_to_images(pptx_path, output_dir):
    abs_pptx = os.path.abspath(pptx_path)
    abs_out = os.path.abspath(output_dir)
    os.makedirs(abs_out, exist_ok=True)
    
    print(f"Opening {abs_pptx}...")
    ppt = win32com.client.Dispatch("PowerPoint.Application")
    # ppt.Visible = 1
    presentation = ppt.Presentations.Open(abs_pptx, WithWindow=False)
    
    print(f"Exporting {len(presentation.Slides)} slides to {abs_out}...")
    # Export method: path, filter name, width, height
    # 1920x1080 for high resolution inspection
    presentation.Export(abs_out, "PNG", 1920, 1080)
    
    presentation.Close()
    ppt.Quit()
    print("Render complete!")

if __name__ == "__main__":
    render_pptx_to_images(r"presentation\GridSentinal_Hackathon_2026.pptx", r"presentation\rendered")
