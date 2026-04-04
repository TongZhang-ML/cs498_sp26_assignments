(TeX-add-style-hook
 "hw"
 (lambda ()
   (TeX-run-style-hooks
    "latex2e"
    "myheaders"
    "instructions"
    "questions/coding_gan"
    "questions/coding_diffusion"
    "article"
    "art10"
    "caption")
   (TeX-add-symbols
    "hwno"
    "duedate"
    "lastduedate"
    "hwc"))
 :latex)

