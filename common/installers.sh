# install_python() {
#   # This function installs CPython.
#   # Usage: install_python BRANCH_TAG TARGET_DIR
#   # Arguments:
#   #  - BRANCH_TAG: Branch or tag of CPython to be checked out.
#   #  - TARGET_DIR: Directory where CPython will be installed. 

#   if [ $# -ne 2 ]; then
#     set +x
#     >&2 echo Usage: install_python BRANCH_TAG TARGET_DIR
#     exit 1
#   fi

#   git clone https://github.com/python/cpython -b $1
#   pushd cpython
#   ./configure --prefix=$2 --enable-optimizations
#   make -j 64
#   make install
#   popd 
# }

install_python() {
  # This function installs CPython.
  # Usage: install_python BRANCH_TAG TARGET_DIR
  # Arguments:
  #  - BRANCH_TAG: Branch or tag of CPython to be checked out.
  #  - TARGET_DIR: Directory where CPython will be installed. 

  if [ $# -ne 2 ]; then
    set +x
    >&2 echo Usage: install_python BRANCH_TAG TARGET_DIR
    exit 1
  fi

  local temp_dir=$(mktemp -d)
  uv python install $1 --install-dir $temp_dir
  local installed_py_dir=$(find $temp_dir -maxdepth 1 -type d -name "cpython-*")
  mkdir $2
  cp -r $installed_py_dir/* $2/
}
