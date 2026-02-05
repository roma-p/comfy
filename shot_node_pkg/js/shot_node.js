import { app } from "/scripts/app.js";

const PLACEHOLDER = "-- Select --";

app.registerExtension({
    name: "shot_node.reactive_dropdown",

    setup() {
        const OriginalContextMenu = LiteGraph.ContextMenu;

        LiteGraph.ContextMenu = function (values, options) {
            // Check if this is a marked sequence widget
            if (values._shotNodeSequence) {
                const xhr = new XMLHttpRequest();
                xhr.open("GET", "/shot_node/sequences", false);
                xhr.send();
                if (xhr.status === 200) {
                    const seqs = [PLACEHOLDER, ...JSON.parse(xhr.responseText)];
                    seqs._shotNodeSequence = true;
                    values.length = 0;
                    values.push(...seqs);
                }
            }
            return new OriginalContextMenu(values, options);
        };

        Object.assign(LiteGraph.ContextMenu, OriginalContextMenu);
        LiteGraph.ContextMenu.prototype = OriginalContextMenu.prototype;
    },

    nodeCreated(node) {
        if (node.comfyClass !== "ShotNode") return;

        const sequenceWidget = node.widgets.find((w) => w.name === "sequence");
        const shotWidget = node.widgets.find((w) => w.name === "shot");
        if (!sequenceWidget || !shotWidget) return;

        // Mark the values array so we can identify it in ContextMenu
        sequenceWidget.options.values._shotNodeSequence = true;

        const originalCallback = sequenceWidget.callback;
        sequenceWidget.callback = async function (value) {
            originalCallback?.call(this, value);

            if (value === PLACEHOLDER) {
                shotWidget.options.values = [PLACEHOLDER];
                shotWidget.value = PLACEHOLDER;
            } else {
                try {
                    const res = await fetch(`/shot_node/shots/${encodeURIComponent(value)}`);
                    if (res.ok) {
                        const shots = await res.json();
                        shotWidget.options.values = [PLACEHOLDER, ...shots];
                        shotWidget.value = PLACEHOLDER;
                    }
                } catch (e) {
                    console.error("Failed to fetch shots:", e);
                }
            }
            app.graph.setDirtyCanvas(true);
        };
    },
});
