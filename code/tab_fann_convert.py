#!/usr/bin/env python

# Parse filename from cmdline
from optparse import OptionParser
parser = OptionParser()
parser.add_option("-i", "--input-file", dest="infile", help="load tab formatted data in FILE", metavar="FILE")
parser.add_option("-o", "--output-file", dest="outfile", help="write converted data to FILE", metavar="FILE")
(options, args) = parser.parse_args()
if None in [options.infile, options.outfile]:
    parser.error("minimum 2 argument required for converting")
print "Converting data from ", options.infile, "to fann format ->",options.outfile,"\n"

import orange

data= orange.ExampleTable(options.infile)

num_attributes = len(data.domain.attributes)

attributes = data.domain.attributes

print data.domain

print data.domain.classVar.name
print data.domain.classVar.values

print num_attributes

num_instances = len(data)

outf = open(options.outfile, "w")

print >>outf,'%s %s %s' % (num_instances, num_attributes, len(data.domain.classVar.values))


attribute_values = {}

class_names = list(data.domain.classVar.values)

for index, attribute in enumerate(attributes):


    attribute_values[attribute] = []

    try:
        values = data.domain[index].values
        for i, value in enumerate(values):
            attribute_values[attribute].append(value)
    except:
        pass


for row in data:
    value_row = []
    for attribute, value in zip(attributes, row):
        if attribute_values[attribute]:
            print attribute_values[attribute], value
            index = attribute_values[attribute].index(value)
            value_row.append(index)
        else:
            value = value.value

            value_row.append(value)

    value_row = [str(a) for a in value_row]
    print >>outf," ".join(value_row)

    class_var_index = class_names.index(row[num_attributes])
    #print >>outf, class_var_index
    output = []
    for class_name in class_names:
        if class_name == row[num_attributes]:
            output.append('1')
        else:
            output.append('0')

    print >>outf, " ".join(output)

