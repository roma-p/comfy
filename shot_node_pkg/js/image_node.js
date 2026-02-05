import { app } from "/scripts/app.js";

const PLACEHOLDER = "-- Select --";

function setupAutocomplete(widget, suggestions) {
    if (widget._hasAutocomplete) return;
    widget._hasAutocomplete = true;
    widget._autocompleteSuggestions = suggestions;

    // Store reference to active menu so we can close it
    let activeMenu = null;

    // Hook into the widget's input element when it's created
    const originalDraw = widget.draw;
    widget.draw = function(ctx, node, width, y, height) {
        originalDraw?.call(this, ctx, node, width, y, height);

        // Find the input element if it exists
        const canvas = ctx.canvas;
        if (canvas && canvas.parentElement) {
            const input = canvas.parentElement.querySelector('input[type="text"], textarea');
            if (input && !input._autocompleteSetup && input === document.activeElement) {
                setupInputAutocomplete(input, widget, node);
            }
        }
    };

    function setupInputAutocomplete(input, widget, node) {
        if (input._autocompleteSetup) return;
        input._autocompleteSetup = true;

        input.addEventListener('input', (e) => {
            const value = e.target.value.toLowerCase();
            if (!value) {
                closeMenu();
                return;
            }

            const filtered = widget._autocompleteSuggestions.filter(s =>
                s.toLowerCase().includes(value)
            );

            if (filtered.length > 0 && filtered[0].toLowerCase() !== value) {
                showSuggestions(filtered, input, widget, node);
            } else {
                closeMenu();
            }
        });

        input.addEventListener('blur', () => {
            setTimeout(closeMenu, 150);
        });
    }

    function showSuggestions(filtered, input, widget, node) {
        closeMenu();

        const rect = input.getBoundingClientRect();

        activeMenu = new LiteGraph.ContextMenu(
            filtered,
            {
                left: rect.left,
                top: rect.bottom,
                callback: (item) => {
                    widget.value = item;
                    input.value = item;
                    widget.callback?.(item);
                    app.graph.setDirtyCanvas(true);
                }
            }
        );
    }

    function closeMenu() {
        if (activeMenu) {
            activeMenu.close?.();
            activeMenu = null;
        }
    }
}

function removeAutocomplete(widget) {
    widget._hasAutocomplete = false;
    widget._autocompleteSuggestions = null;
}

app.registerExtension({
    name: "image_node.dynamic_visibility",

    async nodeCreated(node) {
        if (node.comfyClass !== "ImageNode") return;
        console.log("ImageNode created, setting up dynamic visibility");

        const imageIdWidget = node.widgets.find((w) => w.name === "image_id");
        if (!imageIdWidget) {
            console.error("image_id widget not found");
            return;
        }

        // Fetch all possible fields from API
        let allFields = [];
        try {
            const res = await fetch("/image_node/fields");
            if (res.ok) {
                allFields = await res.json();
                console.log("All fields:", allFields);
            }
        } catch (e) {
            console.error("Failed to fetch image fields:", e);
            return;
        }

        // Log all existing widgets
        console.log("All widgets on node:", node.widgets?.map(w => w.name));

        // Store references to field widgets
        const fieldWidgets = {};
        for (const field of allFields) {
            const widget = node.widgets.find((w) => w.name === field);
            if (widget) {
                fieldWidgets[field] = widget;
                widget._originalComputeSize = widget.computeSize;
            } else {
                console.warn(`Widget not found for field: ${field}`);
            }
        }
        console.log("Field widgets found:", Object.keys(fieldWidgets));

        function hideWidget(widget) {
            widget.hidden = true;
            widget.computeSize = () => [0, -4];
        }

        function showWidget(widget) {
            widget.hidden = false;
            if (widget._originalComputeSize) {
                widget.computeSize = widget._originalComputeSize;
            } else {
                delete widget.computeSize;
            }
        }

        async function updateVisibility(imageId) {
            console.log("updateVisibility called with:", imageId);
            if (imageId === PLACEHOLDER) {
                for (const widget of Object.values(fieldWidgets)) {
                    hideWidget(widget);
                }
            } else {
                try {
                    const res = await fetch(`/image_node/visible_fields/${encodeURIComponent(imageId)}`);
                    if (res.ok) {
                        const visibleFields = await res.json();
                        console.log("Visible fields for", imageId, ":", visibleFields);
                        for (const [field, widget] of Object.entries(fieldWidgets)) {
                            if (visibleFields.includes(field)) {
                                showWidget(widget);
                            } else {
                                hideWidget(widget);
                            }
                        }
                    }
                } catch (e) {
                    console.error("Failed to fetch visible fields:", e);
                }
            }
            node.setSize(node.computeSize());
            app.graph.setDirtyCanvas(true);
        }

        const originalCallback = imageIdWidget.callback;
        imageIdWidget.callback = async function (value) {
            console.log("image_id changed to:", value);
            originalCallback?.call(this, value);
            await updateVisibility(value);
        };

        // Setup task autocomplete when colorspace has value
        const colorspaceWidget = node.widgets.find(w => w.name === "colorspace");

        function updateTaskWidget() {
            const taskWidget = node.widgets.find(w => w.name === "task");
            if (!taskWidget) return;

            const colorspaceValue = colorspaceWidget?.value;
            if (colorspaceValue && colorspaceValue.trim() && colorspaceValue !== PLACEHOLDER) {
                // Colorspace has value - enable autocomplete on task
                setupAutocomplete(taskWidget, ["anim", "comp"]);
            } else {
                // Colorspace empty - remove autocomplete
                removeAutocomplete(taskWidget);
            }
        }

        if (colorspaceWidget) {
            const origColorspaceCallback = colorspaceWidget.callback;
            colorspaceWidget.callback = function(value) {
                origColorspaceCallback?.call(this, value);
                updateTaskWidget();
            };

            // Also listen for direct input changes
            const origOnChange = colorspaceWidget.options?.onChange;
            colorspaceWidget.options = colorspaceWidget.options || {};
            colorspaceWidget.options.onChange = function(value) {
                origOnChange?.call(this, value);
                updateTaskWidget();
            };
        }

        // Initial state
        updateVisibility(imageIdWidget.value);
        updateTaskWidget();
    },
});
