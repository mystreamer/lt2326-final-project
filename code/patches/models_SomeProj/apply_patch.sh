cd ../../models/SomeProj
# dry run
git apply --check ../../patches/models_SomeProj/0001-SomeProj.patch
# full run
git apply ../../patches/models_SomeProj/0001-SomeProj.patch
cd -