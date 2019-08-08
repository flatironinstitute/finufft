import re


decimalMatchString = "\d+\.?\d+" #regular expression to match a decimal number
sciNotString = "(\d*.?\d*e-\d* s)" #regular expression to match a number in scientific notation
wholeNumberMatchString = "\d+" 


#search string needs to have two groupings! (one for everything besides) (time s)
def extractTime(searchString, strOut):
    time = 0
    lineMatch = re.search(searchString,strOut)
    if(lineMatch):
        val = re.search(sciNotString,lineMatch.group(2))
        if(not val):
            val = re.search(decimalMatchString, lineMatch.group(2))
        if(not val):
            val = re.search(wholeNumberMatchString, lineMatch.group(2))
        time = round(float(val.group(0).split('s')[0].strip()),5)
    return time


def sumAllTime(searchString, strOut):
    newVal = 0
    lineMatch = re.findall(searchString,strOut)
    for match in lineMatch:
        val = re.search(sciNotString, match[1])
        if(not val): #search failed, try decimal format 
            val = re.search(decimalMatchString, match[1])
        if(not val):
            val = re.search(wholeNumberMatchString, match[1])
        newVal = newVal + float(val.group(0).split('s')[0].strip()) #trim off " s"
    newVal = round(newVal,5)
    return newVal


