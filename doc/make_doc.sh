rm -f source/wnstatmech.*.rst
mkdir -p source/_static source/_templates
sphinx-apidoc -M -f -n -o source ../wnstatmech
make html
