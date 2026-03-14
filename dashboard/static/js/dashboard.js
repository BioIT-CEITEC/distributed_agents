/**
 * AI Orchestration Metrics Dashboard — Highcharts.js Frontend
 *
 * Architecture:
 *  1. Charts stored in registry — updated via chart.update()
 *  2. Scope selector (Requests / Queries / Traces) sends scope param to API
 *  3. Log file selector — auto-detects and hot-reloads log files
 *  4. Query selector — shows extracted queries with intent classification
 *  5. Highcharts dark theme with enlarged tooltips
 */

// ============================================================
// Color Palette
// ============================================================
const COLORS = {
    purple: '#667eea', purpleA: 'rgba(102, 126, 234, 0.7)',
    cyan:   '#00d2ff', cyanA:   'rgba(0, 210, 255, 0.7)',
    pink:   '#f093fb', pinkA:   'rgba(240, 147, 251, 0.7)',
    green:  '#00e676', greenA:  'rgba(0, 230, 118, 0.7)',
    amber:  '#ffab40', amberA:  'rgba(255, 171, 64, 0.7)',
    red:    '#ff5252', redA:    'rgba(255, 82, 82, 0.7)',
    blue:   '#4facfe', blueA:   'rgba(79, 172, 254, 0.7)',
};
const PALETTE   = [COLORS.purple, COLORS.cyan, COLORS.pink, COLORS.green, COLORS.amber, COLORS.blue, COLORS.red];
const PALETTE_A = [COLORS.purpleA, COLORS.cyanA, COLORS.pinkA, COLORS.greenA, COLORS.amberA, COLORS.blueA, COLORS.redA];

// ============================================================
// Highcharts Global Dark Theme — with LARGER TOOLTIPS
// ============================================================
Highcharts.setOptions({
    chart: {
        backgroundColor: 'transparent',
        style: { fontFamily: '"Inter", sans-serif' },
        animation: { duration: 500 },
    },
    title: { text: null },
    credits: { enabled: false },
    colors: PALETTE,
    tooltip: {
        backgroundColor: 'rgba(14, 14, 36, 0.97)',
        borderColor: 'rgba(102, 126, 234, 0.35)',
        borderRadius: 12,
        padding: 14,
        style: { color: '#d0d0e8', fontSize: '14px', lineHeight: '20px' },
        useHTML: true,
        shadow: { color: 'rgba(0,0,0,0.5)', offsetX: 0, offsetY: 4, width: 8 },
    },
    legend: {
        itemStyle: { color: '#8888a8', fontWeight: '500', fontSize: '11px' },
        itemHoverStyle: { color: '#e8e8f0' },
    },
    xAxis: {
        labels: { style: { color: '#8888a8', fontSize: '10px' } },
        lineColor: 'rgba(100, 100, 180, 0.15)',
        tickColor: 'rgba(100, 100, 180, 0.15)',
        gridLineColor: 'rgba(100, 100, 180, 0.08)',
    },
    yAxis: {
        labels: { style: { color: '#8888a8', fontSize: '10px' } },
        gridLineColor: 'rgba(100, 100, 180, 0.08)',
        title: { style: { color: '#8888a8', fontSize: '11px' } },
    },
    plotOptions: {
        series: {
            animation: { duration: 500 },
            borderWidth: 0,
        },
        pie: {
            borderWidth: 2,
            borderColor: 'rgba(18, 18, 46, 0.75)',
        },
    },
});

// ============================================================
// State
// ============================================================
var charts = {};
var currentScope     = 'requests';
var currentFilterId  = 'all';
var currentQueryId   = 'all';
var filterList       = [];
var queryList        = [];
var filtersPopulated = false;

// ============================================================
// API helper — injects scope and filter params
// ============================================================
async function fetchApi(ep) {
    var url = new URL(ep, window.location.origin);
    url.searchParams.set('scope', currentScope);
    if (currentFilterId !== 'all') {
        url.searchParams.set('request_id', currentFilterId);
    }
    var r = await fetch(url);
    return r.json();
}

// ============================================================
// Smart chart create/update — never destroys on filter changes
// ============================================================
function renderChart(containerId, options) {
    var container = document.getElementById(containerId);
    if (!container) return null;

    var card = container.closest('.chart-card');
    if (card) {
        card.style.opacity = '1';
        card.style.transform = 'none';
    }

    if (charts[containerId]) {
        charts[containerId].update(options, true, true, { duration: 400 });
    } else {
        charts[containerId] = Highcharts.chart(containerId, options);
    }
    return charts[containerId];
}

// ============================================================
// Log File Selector
// ============================================================
async function loadLogFiles() {
    var d = await fetch('/api/logs').then(function(r) { return r.json(); });
    var sel = document.getElementById('logFileSelect');
    sel.innerHTML = '';
    d.files.forEach(function(f) {
        var o = document.createElement('option');
        o.value = f;
        o.textContent = f;
        if (f === d.current) o.selected = true;
        sel.appendChild(o);
    });
    document.getElementById('footerLogName').textContent = d.current || '—';
}

async function switchLogFile(filename) {
    var r = await fetch('/api/load-log', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filename: filename }),
    }).then(function(res) { return res.json(); });

    if (r.error) {
        console.error('[Dashboard] Load log error:', r.error);
        return;
    }

    console.log('[Dashboard] Loaded:', filename, '→', r.spans, 'spans,', r.requests, 'requests');
    document.getElementById('footerLogName').textContent = filename;

    // Destroy all charts on log switch for a clean slate
    Object.keys(charts).forEach(function(id) {
        if (charts[id]) { charts[id].destroy(); charts[id] = null; }
    });
    charts = {};

    // Reset state
    currentFilterId = 'all';
    currentQueryId = 'all';
    filtersPopulated = false;
    queryList = [];

    await loadQueries();
    await renderAll();
}

// ============================================================
// Query Selector
// ============================================================
async function loadQueries() {
    var d = await fetch('/api/queries-list').then(function(r) { return r.json(); });
    queryList = d.queries || [];

    var sel = document.getElementById('querySelect');
    sel.innerHTML = '<option value="all">All Queries</option>';

    queryList.forEach(function(q, i) {
        var o = document.createElement('option');
        o.value = q.request_id;
        // Truncate long queries for dropdown readability
        var label = (i + 1) + '. ' + q.query.substring(0, 80) + (q.query.length > 80 ? '…' : '');
        o.textContent = label;
        o.title = q.query;
        sel.appendChild(o);
    });

    updateQueryInfo();
}

function updateQueryInfo() {
    var textEl = document.getElementById('queryText');
    var badgeEl = document.getElementById('intentBadge');

    if (currentQueryId === 'all') {
        textEl.textContent = queryList.length + ' queries in this session';
        badgeEl.style.display = 'none';
        return;
    }

    var match = queryList.find(function(q) { return q.request_id === currentQueryId; });
    if (match) {
        textEl.textContent = match.query;
        badgeEl.textContent = match.intent.toUpperCase();
        badgeEl.style.display = 'inline-block';
        badgeEl.className = 'intent-badge intent-' + match.intent.toLowerCase();
    }
}

// ============================================================
// Filters
// ============================================================
function populateFilters(items) {
    var sel = document.getElementById('requestFilter');
    sel.innerHTML = '<option value="all">All</option>';
    items.forEach(function(r) {
        var o = document.createElement('option');
        o.value = r.request_id;
        o.textContent = r.request_id.substring(0, 12) + '…';
        sel.appendChild(o);
    });
    filterList = items;
    filtersPopulated = true;
}

function restoreFilterState() {
    document.getElementById('requestFilter').value = currentFilterId;
}

// ============================================================
// KPIs
// ============================================================
async function updateKPIs() {
    var d = await fetchApi('/api/metrics');
    var k = d.kpis;
    document.getElementById('kpiRequests').textContent   = k.total_requests;
    document.getElementById('kpiSpans').textContent      = k.total_spans + ' spans';
    document.getElementById('kpiLatency').textContent    = k.avg_e2e_ms != null ? (k.avg_e2e_ms / 1000).toFixed(1) + 's' : '—';
    document.getElementById('kpiThroughput').textContent  = k.throughput_rps != null ? k.throughput_rps.toFixed(4) : '—';
    document.getElementById('kpiCost').textContent       = k.total_cost_usd != null ? '$' + k.total_cost_usd.toFixed(4) : '—';
    document.getElementById('kpiSuccess').textContent    = k.avg_agent_success != null ? (k.avg_agent_success * 100).toFixed(1) + '%' : '—';
    document.getElementById('kpiPii').textContent        = k.total_pii;
    document.getElementById('kpiErrors').textContent     = k.total_errors + ' errors';
    if (!filtersPopulated) populateFilters(d.requests);
}

// ============================================================
// Chart Renderers
// ============================================================

/**
 * Step-by-Step Latency — horizontal bar chart
 * Shows latency per component with simplified labels
 */
async function renderStepLatency() {
    var d = await fetchApi('/api/latency');
    var s = d.step_latencies || [];

    renderChart('chartStepLatency', {
        chart: { type: 'bar' },
        xAxis: {
            categories: s.map(function(x) { return x.component; }),
            labels: { style: { fontSize: '11px' } },
        },
        yAxis: {
            title: { text: 'Latency (ms)' },
            min: 0,
        },
        tooltip: {
            headerFormat: '<span style="font-size:15px;font-weight:700;color:#f0f0ff">{point.key}</span><br/>',
            pointFormat: '<span style="color:{point.color}">●</span> {series.name}: <b>{point.y:.1f} ms</b><br/>',
        },
        legend: { enabled: false },
        series: [{
            name: 'Latency (ms)',
            data: s.map(function(x) {
                return {
                    y: x.e2e_ms,
                    color: x.component === 'Orchestrator' ? COLORS.purpleA
                         : x.component === 'Home Node' ? COLORS.greenA : COLORS.cyanA,
                };
            }),
            borderRadius: 4,
        }],
    });
}

/**
 * Cost Distribution — donut chart with simplified labels
 */
async function renderCostDonut() {
    var d = await fetchApi('/api/cost');
    var c = d.by_component || {};
    var labels = Object.keys(c);
    var values = Object.values(c);

    var pieData = labels.map(function(label, i) {
        return {
            name: label,
            y: values[i],
            color: PALETTE_A[i % PALETTE_A.length],
        };
    });

    renderChart('chartCostDonut', {
        chart: { type: 'pie' },
        tooltip: {
            pointFormat: '<span style="color:{point.color}">●</span> ${point.y:.5f} (<b>{point.percentage:.1f}%</b>)',
        },
        plotOptions: {
            pie: {
                innerSize: '55%',
                dataLabels: {
                    enabled: true,
                    format: '{point.name}',
                    style: { color: '#8888a8', fontSize: '11px', textOutline: 'none' },
                },
                showInLegend: true,
            },
        },
        series: [{ name: 'Cost (USD)', data: pieData }],
    });
}

/**
 * Latency Timeline — line chart with simplified node labels
 */
async function renderLatencyTimeline() {
    var d = await fetchApi('/api/latency');
    var pts = d.network_latencies || [];

    var byNode = {};
    pts.forEach(function(p) {
        if (!byNode[p.node_id]) byNode[p.node_id] = [];
        byNode[p.node_id].push(p);
    });

    var seriesData = Object.keys(byNode).map(function(nid, i) {
        return {
            name: nid,
            color: PALETTE[i % PALETTE.length],
            marker: {
                enabled: true,
                radius: 5,
                lineWidth: 2,
                lineColor: '#0a0a1a',
                fillColor: PALETTE[i % PALETTE.length],
            },
            data: byNode[nid].map(function(p) {
                return {
                    x: new Date(p.timestamp).getTime(),
                    y: p.latency_ms,
                };
            }).sort(function(a, b) { return a.x - b.x; }),
        };
    });

    renderChart('chartLatencyTimeline', {
        chart: { type: 'line' },
        xAxis: {
            type: 'datetime',
            labels: {
                format: '{value:%H:%M:%S}',
                style: { fontSize: '10px' },
            },
        },
        yAxis: {
            title: { text: 'Latency (ms)' },
            min: 0,
        },
        tooltip: {
            headerFormat: '<span style="font-size:14px;font-weight:700;color:#f0f0ff">{point.x:%H:%M:%S}</span><br/>',
            pointFormat: '<span style="color:{series.color}">●</span> {series.name}: <b>{point.y:.1f} ms</b><br/>',
            shared: true,
        },
        plotOptions: { line: { lineWidth: 2.5 } },
        series: seriesData,
    });
}

/**
 * Agent Success Rate — donut chart
 */
async function renderAgentSuccess() {
    var d = await fetchApi('/api/agents');
    var o = d.overall || { total: 0, successful: 0 };
    var failed = Math.max(0, o.total - o.successful);

    renderChart('chartAgentSuccess', {
        chart: { type: 'pie' },
        tooltip: {
            pointFormat: '<span style="color:{point.color}">●</span> {point.name}: <b>{point.y}</b> ({point.percentage:.1f}%)',
        },
        plotOptions: {
            pie: {
                innerSize: '50%',
                dataLabels: {
                    enabled: true,
                    format: '{point.name}: {point.y}',
                    style: { color: '#8888a8', fontSize: '11px', textOutline: 'none' },
                },
                showInLegend: true,
            },
        },
        series: [{
            name: 'Agents',
            data: [
                { name: 'Successful', y: o.successful, color: COLORS.greenA },
                { name: 'Failed',     y: failed,       color: COLORS.redA },
            ],
        }],
    });
}

/**
 * Token Usage — stacked bar chart with improved spacing
 */
async function renderTokens() {
    var d = await fetchApi('/api/tokens');
    var r = d.per_request || [];

    renderChart('chartTokens', {
        chart: { type: 'column' },
        xAxis: {
            categories: r.map(function(x) { return x.request_id; }),
        },
        yAxis: {
            title: { text: 'Tokens' },
            min: 0,
            stackLabels: {
                enabled: true,
                style: { color: '#8888a8', fontSize: '10px', textOutline: 'none' },
                formatter: function() { return Highcharts.numberFormat(this.total, 0, '.', ','); },
            },
        },
        tooltip: {
            headerFormat: '<span style="font-size:14px;font-weight:700;color:#f0f0ff">{point.key}</span><br/>',
            pointFormat: '<span style="color:{series.color}">●</span> {series.name}: <b>{point.y:,.0f}</b><br/>',
            shared: true,
        },
        plotOptions: {
            column: {
                stacking: 'normal',
                borderRadius: 3,
                pointPadding: 0.15,
                groupPadding: 0.25,
                maxPointWidth: 60,
            },
        },
        series: [
            { name: 'Prompt Tokens',     data: r.map(function(x) { return x.prompt; }),     color: COLORS.purpleA },
            { name: 'Completion Tokens',  data: r.map(function(x) { return x.completion; }), color: COLORS.cyanA },
        ],
    });
}

/**
 * Safety Alerts — donut chart
 */
async function renderSafety() {
    var d = await fetchApi('/api/safety');
    var t = d.totals || { pii: 0, errors: 0, requests_with_pii: 0 };
    var total = currentFilterId === 'all' ? Math.max(filterList.length, 1) : 1;
    var clean = Math.max(0, total - (t.requests_with_pii > 0 ? 1 : 0));

    renderChart('chartSafety', {
        chart: { type: 'pie' },
        tooltip: {
            pointFormat: '<span style="color:{point.color}">●</span> {point.name}: <b>{point.y}</b>',
        },
        plotOptions: {
            pie: {
                innerSize: '50%',
                dataLabels: {
                    enabled: true,
                    format: '{point.name}: {point.y}',
                    style: { color: '#8888a8', fontSize: '11px', textOutline: 'none' },
                },
                showInLegend: true,
            },
        },
        series: [{
            name: 'Safety',
            data: [
                { name: 'PII Detected', y: t.pii,    color: COLORS.amberA },
                { name: 'Errors',       y: t.errors,  color: COLORS.redA },
                { name: 'Clean',        y: clean,      color: COLORS.greenA },
            ],
        }],
    });
}

/**
 * Cost per Request — bar chart
 */
async function renderCostBar() {
    var d = await fetchApi('/api/cost');
    var r = d.per_request || [];

    renderChart('chartCostBar', {
        chart: { type: 'column' },
        xAxis: {
            categories: r.map(function(x) { return x.request_id; }),
        },
        yAxis: {
            title: { text: 'USD' },
            min: 0,
        },
        tooltip: {
            headerFormat: '<span style="font-size:14px;font-weight:700;color:#f0f0ff">{point.key}</span><br/>',
            pointFormat: '<span style="color:{point.color}">●</span> Cost: <b>${point.y:.5f}</b>',
        },
        legend: { enabled: false },
        series: [{
            name: 'Cost (USD)',
            data: r.map(function(x, i) {
                return { y: x.cost_usd, color: PALETTE_A[i % PALETTE_A.length] };
            }),
            borderRadius: 5,
        }],
    });
}

/**
 * Nodes: Queried vs Data — grouped bar chart
 */
async function renderNodes() {
    var d = await fetchApi('/api/agents');
    var r = d.per_request || [];

    renderChart('chartNodes', {
        chart: { type: 'column' },
        xAxis: {
            categories: r.map(function(x) { return x.request_id; }),
        },
        yAxis: {
            title: { text: 'Count' },
            min: 0,
            allowDecimals: false,
        },
        tooltip: {
            headerFormat: '<span style="font-size:14px;font-weight:700;color:#f0f0ff">{point.key}</span><br/>',
            pointFormat: '<span style="color:{series.color}">●</span> {series.name}: <b>{point.y}</b><br/>',
            shared: true,
        },
        plotOptions: {
            column: { borderRadius: 3 },
        },
        series: [
            { name: 'Nodes Queried',  data: r.map(function(x) { return x.nodes_queried; }),   color: COLORS.purpleA },
            { name: 'Nodes w/ Data',  data: r.map(function(x) { return x.nodes_with_data; }), color: COLORS.greenA },
        ],
    });
}

/**
 * Network Latency by Node — bar chart with simplified labels
 */
async function renderNetworkLatency() {
    var d = await fetchApi('/api/latency');
    var nets = d.network_latencies || [];

    var byNode = {};
    nets.forEach(function(n) {
        if (!byNode[n.node_id]) byNode[n.node_id] = [];
        byNode[n.node_id].push(n.latency_ms);
    });
    var labels = Object.keys(byNode);
    var avgs = labels.map(function(k) {
        var a = byNode[k];
        return Math.round((a.reduce(function(s, v) { return s + v; }, 0) / a.length) * 100) / 100;
    });

    renderChart('chartNetworkLatency', {
        chart: { type: 'column' },
        xAxis: {
            categories: labels,
        },
        yAxis: {
            title: { text: 'ms' },
            min: 0,
        },
        tooltip: {
            headerFormat: '<span style="font-size:14px;font-weight:700;color:#f0f0ff">{point.key}</span><br/>',
            pointFormat: '<span style="color:{point.color}">●</span> Avg Latency: <b>{point.y:.1f} ms</b>',
        },
        legend: { enabled: false },
        series: [{
            name: 'Avg Network Latency (ms)',
            data: avgs.map(function(v, i) {
                return { y: v, color: PALETTE_A[i % PALETTE_A.length] };
            }),
            borderRadius: 5,
        }],
    });
}

// ============================================================
// Render all — error-isolated with Promise.allSettled
// ============================================================
async function renderAll() {
    console.log('[Dashboard] renderAll scope=' + currentScope + ' filter=' + currentFilterId);

    // Destroy all existing charts for clean re-render
    Object.keys(charts).forEach(function(id) {
        if (charts[id]) { charts[id].destroy(); charts[id] = null; }
    });
    charts = {};

    await updateKPIs();

    var jobs = [
        ['StepLatency',      renderStepLatency],
        ['CostDonut',        renderCostDonut],
        ['LatencyTimeline',  renderLatencyTimeline],
        ['AgentSuccess',     renderAgentSuccess],
        ['Tokens',           renderTokens],
        ['Safety',           renderSafety],
        ['CostBar',          renderCostBar],
        ['Nodes',            renderNodes],
        ['NetworkLatency',   renderNetworkLatency],
    ];

    await Promise.allSettled(jobs.map(function(pair) {
        return pair[1]().catch(function(e) { console.error('[' + pair[0] + ']', e); });
    }));

    restoreFilterState();
}

// ============================================================
// Event Handlers
// ============================================================

// Log file selector
document.getElementById('logFileSelect').addEventListener('change', function(e) {
    if (e.target.value) switchLogFile(e.target.value);
});

// Scope pills
document.querySelectorAll('.scope-pill').forEach(function(pill) {
    pill.addEventListener('click', function() {
        document.querySelectorAll('.scope-pill').forEach(function(p) { p.classList.remove('active'); });
        pill.classList.add('active');
        currentScope = pill.getAttribute('data-scope');
        currentFilterId = 'all';
        filtersPopulated = false;
        renderAll();
    });
});

// Filter dropdown
document.getElementById('requestFilter').addEventListener('change', function(e) {
    currentFilterId = e.target.value;
    renderAll();
});

// Query selector
document.getElementById('querySelect').addEventListener('change', function(e) {
    currentQueryId = e.target.value;
    if (currentQueryId !== 'all') {
        currentFilterId = currentQueryId;
    } else {
        currentFilterId = 'all';
    }
    updateQueryInfo();
    filtersPopulated = false;
    renderAll();
});

// Refresh button
document.getElementById('refreshBtn').addEventListener('click', function() {
    currentFilterId = 'all';
    currentQueryId = 'all';
    filtersPopulated = false;
    document.getElementById('querySelect').value = 'all';
    updateQueryInfo();
    renderAll();
});

// ============================================================
// Initial load
// ============================================================
document.addEventListener('DOMContentLoaded', async function() {
    await loadLogFiles();
    await loadQueries();
    await renderAll();
});
