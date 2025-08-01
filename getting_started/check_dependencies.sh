for pkg in ffmpeg libsm6 libxext6; do
    if dpkg -s "$pkg" 2>/dev/null | grep -q '^Status: install ok installed'; then
        echo "$pkg : installed"
    else
        echo "$pkg : NOT installed"
    fi
done