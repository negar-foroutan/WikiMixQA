import os
import glob
import cairosvg
import timeout_decorator


@timeout_decorator.timeout(3)
def convert(svg_file, output_file):
    cairosvg.svg2png(url=svg_file, write_to=output_file)


def convert_svg_images(folder_path):
    # Create a pattern to match SVG files
    svg_pattern = os.path.join(folder_path, "*.svg")

    # Use glob to get a list of file paths that match the pattern
    svg_files = glob.glob(svg_pattern)

    # Print the list of SVG files
    for svg_file in svg_files:
        output_file = svg_file.replace(".svg", ".png")
        if os.path.exists(output_file):
            continue
        try:
            print(f"Converting {svg_file} to PNG...")
            convert(svg_file, output_file)
        except Exception as e:
            print(e)
            print(f"Could not convert {svg_file} to PNG!")


# Specify the folder path containing SVG images
folder_path = "data/images_3-10/"

# Call the function to list SVG images in the specified folder
convert_svg_images(folder_path)
