cd ../../models/SomeProj
git format-patch 006f01bc2^..669b3a1b6 --stdout > ../../patches/models_SomeProj/0001-SomeProj.patch
cd -

# for binary and uncommitted files
# git diff > mypatch.patch
# git diff --cached > mypatch.patch
