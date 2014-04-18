#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import posixpath
import urlparse 

BASE_HTML_PATH = os.path.join('output', 'base', 'index.html')
OUTPUT_HTML_DIR = os.path.join('output', 'htmls')

WIDTH_LIMIT = 800

PATCH_HTML = '''<div class="item off" id="%(patch_id)s" style="background: url('%(image_url)s') -%(left_position)dpx -%(top_position)dpx; width: %(size)dpx; height: %(size)dpx; border-width: %(border)dpx"></div>'''

def main():
    parser = argparse.ArgumentParser(description='Generate HTML file')
    parser.add_argument('image_url', type=str)
    parser.add_argument('patch_size', type=int)
    parser.add_argument('num_col_images', type=int)
    parser.add_argument('num_row_images', type=int)
    parser.add_argument('border_width', type=int)
    
    args = parser.parse_args()
    generate(args.image_url, args.patch_size, args.num_col_images, args.num_row_images, args.border_width)

def generate(image_url, patch_size, num_col_images, num_row_images, border_width):

	width = (patch_size + 2 * border_width) * num_col_images
	if width > WIDTH_LIMIT:
		print 'Warning: Interface width exceeds %d px' % (WIDTH_LIMIT)

	patch_htmls = []
	for i in range(num_row_images):
		for j in range(num_col_images):
			patch_id = '%02d_%02d' % (i, j)
			patch_htmls.append(PATCH_HTML % {'patch_id': patch_id, 'image_url': image_url, 'size': patch_size, 'border': border_width, 'top_position': patch_size*i, 'left_position': patch_size*j} )

	output_html_path = os.path.join(OUTPUT_HTML_DIR, '%s.%d.html' % (os.path.splitext(posixpath.basename(urlparse.urlsplit(image_url).path))[0], patch_size))
	html = file(BASE_HTML_PATH, 'r').read()

	html = html.replace('[SIZE]', str(patch_size)).replace('[WIDTH]', str(width)).replace('[BORDER]', str(border_width)).replace('[ITEMS]', '\n'.join(patch_htmls))

	file(output_html_path, 'w').write(html)
	return True

if __name__ == '__main__':
	main()
