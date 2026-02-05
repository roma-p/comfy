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

// Autocomplete search box (same style as VHS)
function searchBox(event, [x, y], node) {
    if (this.prompt) return;
    this.prompt = true;

    let pathWidget = this;
    let dialog = document.createElement("div");
    dialog.className = "litegraph litesearchbox graphdialog rounded";
    dialog.innerHTML = '<span class="name">Path</span> <input autofocus="" type="text" class="value"><button class="rounded">OK</button><div class="helper"></div>';
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

    async function updateOptions() {
        timeout = null;
        let [path, remainder] = path_stem(input.value);

        if (last_path != path) {
            let params = { path: path };
            if (extensions) {
                params.extensions = extensions;
            }
            let optionsURL = api.apiURL("/read_node/getpath?" + new URLSearchParams(params));
            try {
                let resp = await fetch(optionsURL);
                options = await resp.json();
                options.sort();
            } catch (e) {
                options = [];
            }
            last_path = path;
        }

        options_element.innerHTML = "";
        for (let option of options) {
            if (option.startsWith(remainder)) {
                addResult(option, option.endsWith("/"));
            }
        }
    }

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

                ctx.fillStyle = LiteGraph.WIDGET_SECONDARY_TEXT_COLOR;
                ctx.fillText(this.label || this.name, margin * 2, y + H * 0.7);

                ctx.fillStyle = this.value ? LiteGraph.WIDGET_TEXT_COLOR : "#777";
                ctx.textAlign = "right";
                let displayPath = this.value || this.options.placeholder || "";
                if (displayPath.length > 40) {
                    displayPath = "..." + displayPath.slice(-37);
                }
                ctx.fillText(displayPath, widget_width - margin * 2, y + H * 0.7);
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

        previewWidget.imgEl = document.createElement("img");
        previewWidget.imgEl.style["width"] = "100%";
        previewWidget.imgEl.hidden = true;
        previewWidget.imgEl.onload = () => {
            previewWidget.aspectRatio = previewWidget.imgEl.naturalWidth / previewWidget.imgEl.naturalHeight;
            fitHeight(previewNode);
        };

        previewWidget.parentEl.appendChild(previewWidget.videoEl);
        previewWidget.parentEl.appendChild(previewWidget.imgEl);

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
                this.videoEl.src = api.apiURL("/read_node/viewvideo?" + new URLSearchParams(params));
                this.videoEl.hidden = false;
                this.imgEl.hidden = true;
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

app.registerExtension({
    name: "read_node.preview",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "ReadNode") return;

        replacePathWidget(nodeType);
        addVideoPreview(nodeType);
        addLoadWidgetCallback(nodeType, "directory");
        addWidgetCallbacks(nodeType);
    },
});
