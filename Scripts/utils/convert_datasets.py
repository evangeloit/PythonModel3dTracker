# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import xml.etree.ElementTree as et
import copy

# <codecell>

src_tree = et.parse('media/pf/bt_datasets.xml')
src_root = src_tree.getroot()

dest_tree = et.parse('media/pf/bt_datasets_new.xml')
dest_root = dest_tree.getroot()
dest_ds = dest_root.find('dataset')
print dest_root

# <codecell>

def parse_replace(node,source_txt,dest_txt):
    new_node = copy.deepcopy(node)
    for child in new_node.iter():
        if child.text:
            child.text = child.text.replace(source_txt,dest_txt)
    return new_node

def get_did(node):
    did_list = list()
    for ds in node.iter('dataset'):
        did_list.append(ds.find('id').text)
    print did_list
    return did_list
                

# <codecell>

did_list = get_did(src_root)
for did in did_list:
    new_dest_ds = parse_replace(dest_ds,'s09_a08',did)
    dest_root.append(new_dest_ds)
et.dump(dest_root)
dest_tree.write('media/pf/bt_datasets_new1.xml')

# <codecell>


