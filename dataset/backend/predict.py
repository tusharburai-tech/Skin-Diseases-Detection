model = get_model()
status = get_status()

if status == "loading":
    return jsonify({"error": "Model is loading, please wait..."}), 503

if status == "error":
    return jsonify({"error": "Model failed to load"}), 500

if model is None:
    return jsonify({"error": "Model not available"}), 500