import plotly.express as px
import plotly.graph_objects as go


def vol_comparison_chart(results):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=results["date"],
            y=results["implied_vol"] * 100,
            name="Implied vol proxy",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=results["date"],
            y=results["realized_vol"] * 100,
            name="Forward realized vol",
        )
    )
    fig.update_layout(
        xaxis_title=None,
        yaxis_title="Annualized volatility (%)",
        legend_title=None,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def cumulative_pnl_chart(results, side="short"):
    column = "short_gamma_pnl" if side == "short" else "long_gamma_pnl"
    label = f"{side.title()} gamma cumulative P&L proxy"
    cumulative = results[column].cumsum()

    fig = go.Figure(
        go.Scatter(x=results["date"], y=cumulative, name=label)
    )
    fig.update_layout(
        xaxis_title=None,
        yaxis_title="Cumulative proxy P&L ($)",
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def pnl_histogram(results, side="short"):
    column = "short_gamma_pnl" if side == "short" else "long_gamma_pnl"
    fig = px.histogram(results, x=column, nbins=30)
    fig.update_layout(
        xaxis_title="Per-period proxy P&L ($)",
        yaxis_title="Count",
        showlegend=False,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def gex_profile_chart(profile, spot, call_wall, put_wall, zero_gamma):
    fig = go.Figure(
        go.Bar(x=profile["strike"], y=profile["gex"], name="Net GEX")
    )

    fig.add_vline(x=spot, line_dash="solid", annotation_text="Spot")
    if call_wall == call_wall:
        fig.add_vline(x=call_wall, line_dash="dot", annotation_text="Call wall")
    if put_wall == put_wall:
        fig.add_vline(x=put_wall, line_dash="dot", annotation_text="Put wall")
    if zero_gamma == zero_gamma:
        fig.add_vline(
            x=zero_gamma,
            line_dash="dash",
            annotation_text="Zero gamma",
        )

    fig.update_layout(
        xaxis_title="Strike",
        yaxis_title="Signed GEX per 1% move ($)",
        margin=dict(l=10, r=10, t=40, b=10),
        showlegend=False,
    )
    return fig
