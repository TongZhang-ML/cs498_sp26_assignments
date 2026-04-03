(TeX-add-style-hook
 "hw"
 (lambda ()
   (TeX-run-style-hooks
    "latex2e"
    "myheaders"
    "instructions"
    "questions/coding_clip"
    "questions/coding_vlm"
    "article"
    "art10"
    "caption")
   (TeX-add-symbols
    "hwno"
    "duedate"
    "lastduedate"
    "hwc"))
 :latex)

