import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

function chainCallback(object, property, callback) {
    if (object == undefined) {
        console.error("Tried to add callback to non-existant object");
        return;
    }
    if (property in object && object[property]) {
        const callback_orig = object[property];
        object[property] = function () {
            const r = callback_orig.apply(this, arguments);
            return callback.apply(this, arguments) ?? r;
        };
    } else {
        object[property] = callback;
    }
}

function fitHeight(node) {
    node.setSize([node.size[0], node.computeSize([node.size[0], node.size[1]])[1]]);
    node?.graph?.setDirtyCanvas(true);
}

// Path helper - split path into directory and remainder
function path_stem(path) {
    let index = path.lastIndexOf("/");
    if (index < 0) {
        index = path.lastIndexOf("\\");
    }
    if (index < 0) {
        return ["", path];
    }
    return [path.slice(0, index + 1), path.slice(index + 1)];
}

// EXR-only output definitions (layers and cryptomatte)
// These are added AFTER metadata for EXR files
const EXR_ONLY_OUTPUTS = [
    { name: "layers", type: "DICT" },
    { name: "cryptomatte", type: "DICT" },
];

// Base outputs: images, masks, frame_count, metadata (always visible)
// Output order matches Python: IMAGE, MASK, INT, STRING, DICT, DICT
const BASE_OUTPUT_COUNT = 4;

// Track whether EXR outputs are currently shown
const nodeExrState = new WeakMap();

// Detect file type and update output visibility
async function updateOutputVisibility(node, sequencePath) {
    if (!sequencePath) {
        setExrOutputsVisible(node, false);
        return;
    }

    try {
        let detectURL = api.apiURL("/read_node/detect_type?" + new URLSearchParams({ path: sequencePath }));
        let resp = await fetch(detectURL);
        let data = await resp.json();

        setExrOutputsVisible(node, data.type === "exr");
    } catch (e) {
        console.error("Failed to detect file type:", e);
        setExrOutputsVisible(node, true);
    }
}

function setExrOutputsVisible(node, visible) {
    if (!node.outputs) return;

    const currentState = nodeExrState.get(node) || false;
    if (currentState === visible) return;

    nodeExrState.set(node, visible);

    if (visible) {
        // Add EXR-only outputs (layers, cryptomatte) after metadata
        if (node.outputs.length === BASE_OUTPUT_COUNT) {
            for (let output of EXR_ONLY_OUTPUTS) {
                node.addOutput(output.name, output.type);
            }
        }
    } else {
        // Remove EXR-only outputs from the end (indices 5, 4)
        while (node.outputs.length > BASE_OUTPUT_COUNT) {
            const idx = node.outputs.length - 1;
            // Disconnect any links first
            if (node.outputs[idx].links && node.outputs[idx].links.length > 0) {
                node.disconnectOutput(idx);
            }
            node.removeOutput(idx);
        }
    }

    node.setSize(node.computeSize());
    node.graph?.setDirtyCanvas(true);
}

// Autocomplete search box (same style as VHS)
function searchBox(event, [x, y], node) {
    if (this.prompt) return;
    this.prompt = true;

    let pathWidget = this;
    let dialog = document.createElement("div");
    dialog.className = "litegraph litesearchbox graphdialog rounded";
    dialog.innerHTML = '<span class="name">Sequence</span> <input autofocus="" type="text" class="value" placeholder="path/to/image.####.exr"><button class="rounded">OK</button><div class="helper"></div>';
    dialog.close = () => {
        dialog.remove();
    };
    document.body.append(dialog);

    if (app.canvas.ds.scale > 1) {
        dialog.style.transform = "scale(" + app.canvas.ds.scale + ")";
    }

    var input = dialog.querySelector(".value");
    var options_element = dialog.querySelector(".helper");
    input.value = pathWidget.value || "";

    var timeout = null;
    let last_path = null;
    let extensions = pathWidget.options.vhs_path_extensions;
    let options = [];

    input.addEventListener("keydown", (e) => {
        dialog.is_modified = true;
        if (e.keyCode == 27) {
            // ESC
            dialog.close();
            pathWidget.prompt = false;
        } else if (e.keyCode == 13 && e.target.localName != "textarea") {
            // Enter
            pathWidget.value = input.value;
            if (pathWidget.callback) {
                pathWidget.callback(pathWidget.value);
            }
            dialog.close();
            pathWidget.prompt = false;
        } else {
            if (e.keyCode == 9) {
                // Tab - autocomplete
                if (options_element.firstChild) {
                    input.value = last_path + options_element.firstChild.innerText;
                }
                e.preventDefault();
                e.stopPropagation();
            }
            if (timeout) clearTimeout(timeout);
            timeout = setTimeout(updateOptions, 10);
            return;
        }
        e.preventDefault();
        e.stopPropagation();
    });

    var button = dialog.querySelector("button");
    button.addEventListener("click", (e) => {
        pathWidget.value = input.value;
        if (pathWidget.callback) {
            pathWidget.callback(pathWidget.value);
        }
        node.graph?.setDirtyCanvas(true);
        dialog.close();
        pathWidget.prompt = false;
    });

    // Position at cursor
    var rect = app.canvas.canvas.getBoundingClientRect();
    var offsetx = -20;
    var offsety = -20;
    if (rect) {
        offsetx -= rect.left;
        offsety -= rect.top;
    }
    if (event) {
        dialog.style.left = event.clientX + offsetx + "px";
        dialog.style.top = event.clientY + offsety + "px";
    }

    function addResult(name, isDir) {
        let el = document.createElement("div");
        el.innerText = name;
        el.className = "litegraph lite-search-item";
        if (isDir) {
            el.className += " is-dir";
            el.addEventListener("click", () => {
                input.value = last_path + name;
                if (timeout) clearTimeout(timeout);
                timeout = setTimeout(updateOptions, 10);
            });
        } else {
            el.addEventListener("click", () => {
                pathWidget.value = last_path + name;
                if (pathWidget.callback) {
                    pathWidget.callback(pathWidget.value);
                }
                dialog.close();
                pathWidget.prompt = false;
            });
        }
        options_element.appendChild(el);
    }

    // Check if a filename belongs to a sequence
    function isSequenceFile(filename, sequences) {
        for (let seq of sequences) {
            // Build regex from sequence pattern: test.####.png -> test.\d{4}.png
            let escapedPrefix = seq.pattern.split(/#+/)[0].replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
            let escapedSuffix = seq.pattern.split(/#+/).slice(1).join('').replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
            let regex = new RegExp(`^${escapedPrefix}\\d{${seq.padding},}${escapedSuffix}$`);
            if (regex.test(filename)) {
                return true;
            }
        }
        return false;
    }

    async function updateOptions() {
        timeout = null;
        let [path, remainder] = path_stem(input.value);

        let sequences = [];
        let filteredOptions = [];

        if (last_path != path) {
            let params = { path: path };
            if (extensions) {
                params.extensions = extensions;
            }

            // Fetch directory contents and sequences in parallel
            try {
                let [dirResp, seqResp] = await Promise.all([
                    fetch(api.apiURL("/read_node/getpath?" + new URLSearchParams(params))),
                    path ? fetch(api.apiURL("/read_node/detect_sequences?" + new URLSearchParams({ path: path }))) : Promise.resolve(null)
                ]);

                options = await dirResp.json();
                options.sort();

                if (seqResp) {
                    let seqData = await seqResp.json();
                    sequences = seqData.sequences || [];
                }
            } catch (e) {
                options = [];
                sequences = [];
            }

            // Filter out files that belong to sequences
            filteredOptions = options.filter(opt => {
                // Keep directories
                if (opt.endsWith("/")) return true;
                // Filter out sequence member files
                return !isSequenceFile(opt, sequences);
            });

            last_path = path;
            // Cache for re-rendering
            last_sequences = sequences;
            last_filtered = filteredOptions;
        } else {
            sequences = last_sequences || [];
            filteredOptions = last_filtered || options;
        }

        options_element.innerHTML = "";

        // Show sequence patterns first (highlighted)
        for (let seq of sequences) {
            // Check if pattern matches remainder filter
            if (seq.pattern.startsWith(remainder)) {
                let el = document.createElement("div");
                el.innerText = `${seq.pattern} (${seq.frame_count} frames)`;
                el.className = "litegraph lite-search-item sequence-pattern";
                el.style.color = "#8cf";
                el.addEventListener("click", () => {
                    pathWidget.value = seq.full_path;
                    if (pathWidget.callback) {
                        pathWidget.callback(pathWidget.value);
                    }
                    dialog.close();
                    pathWidget.prompt = false;
                });
                options_element.appendChild(el);
            }
        }

        // Show remaining files (directories + non-sequence files)
        for (let option of filteredOptions) {
            if (option.startsWith(remainder)) {
                addResult(option, option.endsWith("/"));
            }
        }
    }

    let last_sequences = [];
    let last_filtered = [];

    setTimeout(() => {
        input.focus();
        updateOptions();
    }, 10);
}

// Custom path widget with autocomplete
function createPathWidget(node, inputName, inputData) {
    let w = {
        name: inputName,
        type: "customtext",
        value: inputData[1]?.default || "",
        options: inputData[1] || {},
        draw: function (ctx, node, widget_width, y, H) {
            const show_text = app.canvas.ds.scale >= 0.5;
            const margin = 15;
            ctx.textAlign = "left";
            ctx.strokeStyle = LiteGraph.WIDGET_OUTLINE_COLOR;
            ctx.fillStyle = LiteGraph.WIDGET_BGCOLOR;
            ctx.beginPath();
            if (show_text) {
                ctx.roundRect(margin, y, widget_width - margin * 2, H, [H * 0.5]);
            } else {
                ctx.rect(margin, y, widget_width - margin * 2, H);
            }
            ctx.fill();
            if (show_text && !this.disabled) {
                ctx.stroke();
                ctx.save();
                ctx.beginPath();
                ctx.rect(margin, y, widget_width - margin * 2, H);
                ctx.clip();

                // Display path or placeholder (left-aligned)
                ctx.fillStyle = this.value ? LiteGraph.WIDGET_TEXT_COLOR : "#777";
                ctx.textAlign = "left";
                let displayPath = this.value || this.options.placeholder || "";
                // Truncate from start if too long
                const maxChars = Math.floor((widget_width - margin * 4) / 7);
                if (displayPath.length > maxChars) {
                    displayPath = "..." + displayPath.slice(-(maxChars - 3));
                }
                ctx.fillText(displayPath, margin * 2, y + H * 0.7);
                ctx.restore();
            }
        },
        mouse: searchBox,
        computeSize: function () {
            return [200, 20];
        },
    };
    return w;
}

function addVideoPreview(nodeType) {
    chainCallback(nodeType.prototype, "onNodeCreated", function () {
        var element = document.createElement("div");
        const previewNode = this;

        var previewWidget = this.addDOMWidget("videopreview", "preview", element, {
            serialize: false,
            hideOnZoom: false,
            getValue() {
                return element.value;
            },
            setValue(v) {
                element.value = v;
            },
        });

        previewWidget.computeSize = function (width) {
            if (this.aspectRatio && !this.parentEl.hidden) {
                let height = (previewNode.size[0] - 20) / this.aspectRatio + 10;
                if (!(height > 0)) {
                    height = 0;
                }
                this.computedHeight = height + 10;
                return [width, height];
            }
            return [width, -4];
        };

        previewWidget.value = { hidden: false, paused: false, params: {} };

        previewWidget.parentEl = document.createElement("div");
        previewWidget.parentEl.className = "read_preview";
        previewWidget.parentEl.style["width"] = "100%";
        element.appendChild(previewWidget.parentEl);

        previewWidget.videoEl = document.createElement("video");
        previewWidget.videoEl.controls = false;
        previewWidget.videoEl.loop = true;
        previewWidget.videoEl.muted = true;
        previewWidget.videoEl.style["width"] = "100%";
        previewWidget.videoEl.addEventListener("loadedmetadata", () => {
            previewWidget.aspectRatio = previewWidget.videoEl.videoWidth / previewWidget.videoEl.videoHeight;
            fitHeight(previewNode);
        });
        previewWidget.videoEl.addEventListener("error", () => {
            previewWidget.parentEl.hidden = true;
            fitHeight(previewNode);
        });

        previewWidget.parentEl.appendChild(previewWidget.videoEl);

        var timeout = null;

        this.updateParameters = (params, force_update) => {
            if (!previewWidget.value.params) {
                if (typeof previewWidget.value != "object") {
                    previewWidget.value = { hidden: false, paused: false };
                }
                previewWidget.value.params = {};
            }
            if (!Object.entries(params).some(([k, v]) => previewWidget.value.params[k] !== v)) {
                return;
            }
            Object.assign(previewWidget.value.params, params);
            if (timeout) {
                clearTimeout(timeout);
            }
            if (force_update) {
                previewWidget.updateSource();
            } else {
                timeout = setTimeout(() => previewWidget.updateSource(), 100);
            }
        };

        previewWidget.updateSource = function () {
            if (this.value.params == undefined) {
                return;
            }
            let params = {};
            Object.assign(params, this.value.params);
            params.timestamp = Date.now();
            this.parentEl.hidden = this.value.hidden;

            if (params.format == "folder" && params.filename) {
                this.videoEl.autoplay = !this.value.paused && !this.value.hidden;
                let target_width = (previewNode.size[0] - 20) * 2 || 256;
                if (target_width < 256) {
                    target_width = 256;
                }
                params.force_size = target_width + "x?";
                this.videoEl.src = api.apiURL("/read_node/animated_preview?" + new URLSearchParams(params));
                this.videoEl.hidden = false;
            }
        };

        previewWidget.callback = previewWidget.updateSource;
    });
}

function addLoadWidgetCallback(nodeType, widgetName) {
    chainCallback(nodeType.prototype, "onNodeCreated", function () {
        const pathWidget = this.widgets.find((w) => w.name === widgetName);
        const node = this;

        chainCallback(pathWidget, "callback", (value) => {
            if (!value) {
                return;
            }
            let params = { filename: value, type: "path", format: "folder" };
            node.updateParameters(params, true);

            // Update output visibility based on file type (works with sequence patterns)
            updateOutputVisibility(node, value);
        });
    });
}

function addWidgetCallbacks(nodeType) {
    chainCallback(nodeType.prototype, "onNodeCreated", function () {
        const node = this;

        function update(key) {
            return function (value) {
                let params = {};
                params[key] = this.value;
                node?.updateParameters(params);
            };
        }

        let widgetMap = {
            image_load_cap: "image_load_cap",
            skip_first_images: "skip_first_images",
            select_every_nth: "select_every_nth",
        };

        for (let widget of this.widgets) {
            if (widget.name in widgetMap) {
                chainCallback(widget, "callback", update(widgetMap[widget.name]));
            }
            if (widget.type != "button") {
                widget.callback?.(widget.value);
            }
        }
    });
}

// Replace text widget with path widget
function replacePathWidget(nodeType) {
    chainCallback(nodeType.prototype, "onNodeCreated", function () {
        let new_widgets = [];
        for (let w of this.widgets) {
            let input = this.constructor.nodeData.input;
            let config = input?.required?.[w.name] ?? input?.optional?.[w.name];
            if (config && w?.type == "text" && config[1]?.vhs_path_extensions !== undefined) {
                new_widgets.push(createPathWidget(this, w.name, ["STRING", config[1]]));
            } else {
                new_widgets.push(w);
            }
        }
        this.widgets = new_widgets;
    });
}

// Initialize output visibility on node creation
function initializeOutputVisibility(nodeType) {
    chainCallback(nodeType.prototype, "onNodeCreated", function () {
        const node = this;

        // Remove EXR outputs by default (will be added when EXR sequence is selected)
        setTimeout(() => {
            // Mark as showing EXR so setExrOutputsVisible will remove them
            nodeExrState.set(node, true);
            setExrOutputsVisible(node, false);
        }, 50);
    });

    // Also check visibility when node is configured (e.g., loading from workflow)
    chainCallback(nodeType.prototype, "onConfigure", function (info) {
        const node = this;
        const pathWidget = this.widgets?.find((w) => w.name === "sequence_path");

        // Delay to ensure node is fully configured
        setTimeout(() => {
            if (pathWidget?.value) {
                updateOutputVisibility(node, pathWidget.value);
            } else {
                // Mark as showing EXR so setExrOutputsVisible will remove them
                nodeExrState.set(node, true);
                setExrOutputsVisible(node, false);
            }
        }, 100);
    });
}

app.registerExtension({
    name: "read_node.preview",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "ReadNode") return;

        replacePathWidget(nodeType);
        addVideoPreview(nodeType);
        addLoadWidgetCallback(nodeType, "sequence_path");
        addWidgetCallbacks(nodeType);
        initializeOutputVisibility(nodeType);
    },
});
