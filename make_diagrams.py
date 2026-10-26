"""Generates assets/use_case_diagram.png and assets/decision_workflow.png (needs graphviz)."""
import graphviz

def usecase():
    """Hand-laid-out UML use case diagram (matplotlib) so lines stay readable."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse, FancyBboxPatch, Rectangle

    fig, ax = plt.subplots(figsize=(11, 8.2))
    ax.set_xlim(0, 11); ax.set_ylim(0, 8.2); ax.axis("off")
    ax.add_patch(Rectangle((2.6, 0.2), 5.8, 7.8, fill=False, ec="#555", lw=1.5))
    ax.text(5.5, 7.78, "Facility Reservation Monitoring System (Streamlit app)",
            ha="center", va="center", fontsize=11, weight="bold")
    uc = [("u1", "Submit reservation\n(existing booking system)"),
          ("u2", "Upload & validate CSV"),
          ("u3", "Explore data (EDA)"),
          ("u4", "Predict cancellation /\nno-show risk"),
          ("u7", "View model metrics\n& limitations"),
          ("u5", "View risk dashboard\n& review queue"),
          ("u6", "Record final follow-up\ndecision"),
          ("u8", "Retrain / replace model\n(train_model.py)")]
    ys = {k: 7.1 - i * 0.88 for i, (k, _) in enumerate(uc)}
    X = 5.5
    actors = {"req": (1.2, 6.6, "Requester\n(student, faculty,\nstaff, external)"),
              "ro": (1.2, 4.6, "Reservation\nOfficer"),
              "fa": (1.2, 2.2, "Facility\nAdministrator"),
              "da": (9.8, 3.6, "Data Analyst /\nSystem Admin")}
    links = {"req": ["u1"], "ro": ["u2", "u4", "u5"],
             "fa": ["u2", "u3", "u4", "u7", "u5", "u6"],
             "da": ["u2", "u3", "u7", "u8"]}
    colors = {"req": "#999", "ro": "#2b7bba", "fa": "#c0392b", "da": "#2e8b57"}
    for a, us in links.items():
        ax0, ay0, _ = actors[a]
        for u in us:
            ex = 2.6 if a != "da" else 8.4
            ax.plot([ax0 + (0.8 if a != "da" else -0.8), X + (-1.6 if a != "da" else 1.6)],
                    [ay0, ys[u]], color=colors[a], lw=1.2, zorder=1)
    for k, label in uc:
        ax.add_patch(Ellipse((X, ys[k]), 3.2, 0.68, fc="#dae8fc", ec="#333", zorder=2))
        ax.text(X, ys[k], label, ha="center", va="center", fontsize=8.5, zorder=3)
    for a, (x, y, label) in actors.items():
        ax.add_patch(FancyBboxPatch((x - 0.8, y - 0.45), 1.6, 0.9, boxstyle="round,pad=0.05",
                                    fc="#fff2cc", ec=colors[a], lw=2, zorder=3))
        ax.text(x, y, label, ha="center", va="center", fontsize=8.5, weight="bold", zorder=4)
    for a, b in (("u4", "u7"), ("u6", "u5")):
        ax.annotate("", xy=(X, ys[b] + 0.34), xytext=(X, ys[a] - 0.34),
                    arrowprops=dict(arrowstyle="->", ls="--", color="#333"), zorder=4)
        ax.text(X + 0.08, (ys[a] + ys[b]) / 2, "«include»", fontsize=8, style="italic", zorder=5)
    ax.text(5.5, 0.05, "Line colour = actor.  Only Facility Administrator can record final decisions.",
            ha="center", fontsize=8.5, style="italic")
    plt.savefig("assets/use_case_diagram.png", dpi=160, bbox_inches="tight")
    plt.close()

def workflow():
    g = graphviz.Digraph("wf", format="png")
    g.attr(rankdir="TB", dpi="160", fontname="Helvetica", nodesep="0.35")
    box = dict(shape="box", style="rounded,filled", fillcolor="#dae8fc", fontname="Helvetica")
    g.node("a", "Reservation records\n(CSV upload / bundled data)", **box)
    g.node("b", "Schema validation\n(columns, types, categories)", **box)
    g.node("e", "Show error message;\nrequest corrected file", shape="box", style="rounded,filled", fillcolor="#f8cecc")
    g.node("c", "Cleaning (midterm rules):\nremove duplicates, fix invalid\nvalues, median imputation", **box)
    g.node("d", "Saved pipeline\n(OrdinalEncoder + Decision Tree)\nbooking-time features only", shape="box", style="rounded,filled", fillcolor="#d5e8d4")
    g.node("f", "Risk score P(Not Completed)\n+ High / Medium / Low", **box)
    g.node("g", "Review queue &\nrecommended follow-up", **box)
    g.node("h", "Facility Administrator\nreviews each case", shape="box", style="filled", fillcolor="#fff2cc")
    g.node("i", "Final human decision recorded\n(reviewer, note, timestamp,\nmodel version)", shape="box", style="filled", fillcolor="#fff2cc")
    g.node("j", "Outcome later observed\n-> feed next retraining", shape="box", style="dashed")
    g.node("q", "Valid?", shape="diamond", style="filled", fillcolor="#ffffff")
    g.edge("a", "b"); g.edge("b", "q"); g.edge("q", "e", label="no"); g.edge("q", "c", label="yes")
    g.edge("c", "d"); g.edge("d", "f"); g.edge("f", "g"); g.edge("g", "h"); g.edge("h", "i")
    g.edge("i", "j", style="dashed")
    g.render("assets/decision_workflow", cleanup=True)

if __name__ == "__main__":
    usecase(); workflow()
