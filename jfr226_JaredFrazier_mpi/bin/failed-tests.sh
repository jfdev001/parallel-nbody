#/usr/bin/bash
# @brief Print the file names in the dir for which status of tests fails
for file in tests/*_mynbody.test.out;
do
    grep "STATUS=FAILED" $file && echo $file;
done;