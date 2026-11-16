import { useMemo, useState } from "react";
import "./App.css";

const metrics = {
  flows: "703,240",
  attacks: "286,951",
  attackRate: "41.08%",
  recall: "99.32%",
  prAuc: "0.9983",
};

const experiments = [
  { name: "GraphShield — directed, 5 min", pr: 0.9287, f1: 0.9687, tag: "Reference" },
  { name: "GraphShield — 1 min", pr: 0.9203, f1: 0.9150, tag: "Temporal" },
  { name: "GraphShield — 10 min", pr: 0.9126, f1: 0.9577, tag: "Temporal" },
  { name: "GraphShield — undirected", pr: 0.9222, f1: 0.9268, tag: "Structure" },
  { name: "GraphShield — 64 features", pr: 0.9135, f1: 0.9618, tag: "Ablation" },
];

const alerts = [
  { id: "AL-1042", type: "DDoS", source: "10.24.18.41", dest: "172.16.0.14", confidence: "99.99%", severity: "HIGH" },
  { id: "AL-1041", type: "PortScan", source: "10.24.7.82", dest: "172.16.0.29", confidence: "99.98%", severity: "HIGH" },
  { id: "AL-1040", type: "DDoS", source: "10.24.19.17", dest: "172.16.0.11", confidence: "99.97%", severity: "HIGH" },
  { id: "AL-1039", type: "Suspicious flow", source: "10.24.8.12", dest: "172.16.0.65", confidence: "98.41%", severity: "MEDIUM" },
];

const attackFamilies = [
  { name: "DDoS", value: 128027, recall: "99.95%" },
  { name: "PortScan", value: 158930, recall: "99.82%" },
  { name: "Bot", value: 1966, recall: "0.00%" },
];

const timeline = [9, 12, 18, 14, 26, 38, 32, 46, 42, 63, 57, 71, 66, 82, 77, 91, 74, 88];

function Sparkline() {
  const max = Math.max(...timeline);
  const points = timeline
    .map((v, i) => `${(i / (timeline.length - 1)) * 100},${100 - (v / max) * 86}`)
    .join(" ");

  return (
    <svg className="sparkline" viewBox="0 0 100 100" preserveAspectRatio="none">
      <polyline points={points} fill="none" stroke="currentColor" strokeWidth="2.4" />
      {timeline.map((v, i) => (
        <circle
          key={i}
          cx={(i / (timeline.length - 1)) * 100}
          cy={100 - (v / max) * 86}
          r="1.5"
          fill="currentColor"
        />
      ))}
    </svg>
  );
}

function NetworkGraph() {
  return (
    <div className="network-canvas">
      <div className="network-grid" />
      <svg viewBox="0 0 760 340" className="network-svg">
        <defs>
          <filter id="glow">
            <feGaussianBlur stdDeviation="5" result="coloredBlur" />
            <feMerge>
              <feMergeNode in="coloredBlur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>

        <line x1="110" y1="80" x2="280" y2="150" className="edge" />
        <line x1="110" y1="80" x2="285" y2="270" className="edge" />
        <line x1="280" y1="150" x2="470" y2="105" className="edge" />
        <line x1="280" y1="150" x2="510" y2="235" className="edge attack-edge" />
        <line x1="285" y1="270" x2="510" y2="235" className="edge" />
        <line x1="470" y1="105" x2="650" y2="165" className="edge" />
        <line x1="510" y1="235" x2="650" y2="165" className="edge attack-edge" />

        <g className="node">
          <circle cx="110" cy="80" r="18" />
          <text x="110" y="84" textAnchor="middle">A</text>
        </g>
        <g className="node">
          <circle cx="280" cy="150" r="22" filter="url(#glow)" />
          <text x="280" y="154" textAnchor="middle">B</text>
        </g>
        <g className="node">
          <circle cx="285" cy="270" r="18" />
          <text x="285" y="274" textAnchor="middle">C</text>
        </g>
        <g className="node">
          <circle cx="470" cy="105" r="18" />
          <text x="470" y="109" textAnchor="middle">D</text>
        </g>
        <g className="node suspicious">
          <circle cx="510" cy="235" r="27" filter="url(#glow)" />
          <text x="510" y="239" textAnchor="middle">E</text>
        </g>
        <g className="node">
          <circle cx="650" cy="165" r="18" />
          <text x="650" y="169" textAnchor="middle">F</text>
        </g>
      </svg>

      <div className="graph-tooltip">
        <span className="status-dot danger" />
        <div>
          <strong>Node E · HIGH RISK</strong>
          <small>12,442 flows · 99.97% confidence</small>
        </div>
      </div>
    </div>
  );
}

function Overview() {
  return (
    <>
      <section className="page-head">
        <div>
          <div className="eyebrow">SECURITY OPERATIONS CENTER</div>
          <h1>Threat Overview</h1>
          <p>Temporal graph intelligence over network-flow interactions.</p>
        </div>
        <div className="dataset-pill">
          <span className="status-dot live" />
          <span>JULY 7 EXPLORATORY DATA</span>
        </div>
      </section>

      <section className="stats-grid">
        <Metric label="Total Flows" value={metrics.flows} icon="↗" />
        <Metric label="Detected Attacks" value={metrics.attacks} icon="⚠" tone="danger" />
        <Metric label="Attack Rate" value={metrics.attackRate} icon="%" />
        <Metric label="Detection Recall" value={metrics.recall} icon="✓" tone="success" />
        <Metric label="PR-AUC" value={metrics.prAuc} icon="◎" tone="accent" />
      </section>

      <section className="main-grid">
        <Panel
          title="Attack Activity"
          subtitle="Temporal activity signal · 01:00–12:59"
          action="1 min / 5 min / 10 min"
          className="large"
        >
          <div className="chart-legend">
            <span><i className="legend-line" /> Attack intensity</span>
            <span>Peak activity: 11:00</span>
          </div>
          <div className="activity-chart">
            <div className="chart-y"><span>100</span><span>75</span><span>50</span><span>25</span><span>0</span></div>
            <div className="chart-area">
              <div className="chart-grid-lines">
                <span /><span /><span /><span /><span />
              </div>
              <Sparkline />
              <div className="chart-x"><span>01:00</span><span>04:00</span><span>07:00</span><span>10:00</span><span>12:59</span></div>
            </div>
          </div>
        </Panel>

        <Panel title="Attack Families" subtitle="Later-period family breakdown">
          <div className="family-list">
            {attackFamilies.map((item, idx) => (
              <div className="family-row" key={item.name}>
                <div className="family-name">
                  <span className={`family-dot dot-${idx}`} />
                  <span>{item.name}</span>
                </div>
                <strong>{item.recall}</strong>
                <span className="family-count">{item.value.toLocaleString()}</span>
              </div>
            ))}
          </div>
          <div className="family-footnote">
            Bot traffic requires separate attention in this evaluation.
          </div>
        </Panel>
      </section>

      <section className="main-grid">
        <Panel
          title="Temporal Network Graph"
          subtitle="Interactive endpoint relationships"
          action="Snapshot · 08:35"
          className="large graph-panel"
        >
          <NetworkGraph />
        </Panel>

        <Panel title="Recent Alerts" subtitle="Highest-confidence detections" action="View all">
          <div className="alerts-stack">
            {alerts.slice(0, 3).map((alert) => (
              <div className="alert-card" key={alert.id}>
                <div className="alert-icon">!</div>
                <div className="alert-main">
                  <div className="alert-top">
                    <strong>{alert.type}</strong>
                    <span className="severity">{alert.severity}</span>
                  </div>
                  <span>{alert.source} → {alert.dest}</span>
                  <small>{alert.confidence} confidence</small>
                </div>
              </div>
            ))}
          </div>
        </Panel>
      </section>
    </>
  );
}

function GraphExplorer() {
  const [snapshot, setSnapshot] = useState("08:35");
  const [risk, setRisk] = useState("All");

  return (
    <>
      <section className="page-head">
        <div>
          <div className="eyebrow">GRAPH ANALYSIS</div>
          <h1>Graph Explorer</h1>
          <p>Inspect temporal network structure and endpoint behavior.</p>
        </div>
      </section>

      <div className="explorer-controls panel">
        <label>
          Snapshot
          <select value={snapshot} onChange={(e) => setSnapshot(e.target.value)}>
            <option>08:35</option>
            <option>09:10</option>
            <option>10:25</option>
            <option>11:40</option>
          </select>
        </label>
        <label>
          Risk filter
          <select value={risk} onChange={(e) => setRisk(e.target.value)}>
            <option>All</option>
            <option>High</option>
            <option>Medium</option>
          </select>
        </label>
        <div className="explorer-note">
          <span className="status-dot live" />
          <span>11 node-history features · 66 flow features</span>
        </div>
      </div>

      <section className="main-grid">
        <Panel title={`Network Snapshot · ${snapshot}`} subtitle="Source → destination relationships" className="large graph-panel">
          <NetworkGraph />
        </Panel>

        <Panel title="Selected Entity" subtitle="Representative high-risk node">
          <div className="entity-card">
            <div className="entity-avatar">E</div>
            <div>
              <strong>10.24.19.17</strong>
              <span>Network entity</span>
            </div>
            <div className="risk-badge">HIGH RISK</div>
          </div>
          <div className="entity-stats">
            <div><span>Flows</span><strong>12,442</strong></div>
            <div><span>Unique destinations</span><strong>54</strong></div>
            <div><span>Outgoing ports</span><strong>17</strong></div>
            <div><span>Protocols</span><strong>3</strong></div>
          </div>
          <div className="risk-meter">
            <div className="meter-head"><span>Risk score</span><strong>99.97%</strong></div>
            <div className="meter"><span style={{ width: "99.97%" }} /></div>
          </div>
        </Panel>
      </section>
    </>
  );
}

function Alerts() {
  return (
    <>
      <section className="page-head">
        <div>
          <div className="eyebrow">INCIDENT RESPONSE</div>
          <h1>Alerts</h1>
          <p>Frozen-threshold detections generated by GraphShield.</p>
        </div>
      </section>

      <Panel title="Detection Queue" subtitle="Sorted by severity and confidence">
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>ID</th>
                <th>Threat</th>
                <th>Source</th>
                <th>Destination</th>
                <th>Confidence</th>
                <th>Severity</th>
              </tr>
            </thead>
            <tbody>
              {alerts.map((alert) => (
                <tr key={alert.id}>
                  <td className="mono">{alert.id}</td>
                  <td><strong>{alert.type}</strong></td>
                  <td className="mono">{alert.source}</td>
                  <td className="mono">{alert.dest}</td>
                  <td>{alert.confidence}</td>
                  <td><span className={`severity-pill ${alert.severity.toLowerCase()}`}>{alert.severity}</span></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>
    </>
  );
}

function Experiments() {
  return (
    <>
      <section className="page-head">
        <div>
          <div className="eyebrow">RESEARCH MODE</div>
          <h1>Experiment Lab</h1>
          <p>Validation-first comparison of GraphShield configurations.</p>
        </div>
      </section>

      <section className="stats-grid four">
        <Metric label="Reference Validation PR-AUC" value="0.9287" icon="◎" tone="accent" />
        <Metric label="Reference Validation F1" value="0.9687" icon="✓" tone="success" />
        <Metric label="GNN Parameters" value="27,041" icon="◫" />
        <Metric label="Temporal Windows Tested" value="3" icon="◷" />
      </section>

      <Panel title="Controlled Experiments" subtitle="Validation metrics are the primary comparison evidence">
        <div className="experiment-list">
          {experiments.map((item) => (
            <div className="experiment-row" key={item.name}>
              <div>
                <strong>{item.name}</strong>
                <span>{item.tag}</span>
              </div>
              <div className="experiment-metric">
                <label>PR-AUC</label>
                <strong>{item.pr.toFixed(4)}</strong>
                <div className="mini-meter"><span style={{ width: `${item.pr * 100}%` }} /></div>
              </div>
              <div className="experiment-metric">
                <label>F1</label>
                <strong>{item.f1.toFixed(4)}</strong>
                <div className="mini-meter"><span style={{ width: `${item.f1 * 100}%` }} /></div>
              </div>
            </div>
          ))}
        </div>
      </Panel>

      <div className="research-note">
        <div className="note-icon">i</div>
        <div>
          <strong>Methodology note</strong>
          <p>July 7 is shown separately as an exploratory later-period evaluation because multiple configurations were inspected on that date.</p>
        </div>
      </div>
    </>
  );
}

function Metric({ label, value, icon, tone = "" }) {
  return (
    <div className={`metric-card ${tone}`}>
      <div className="metric-top">
        <span>{label}</span>
        <span className="metric-icon">{icon}</span>
      </div>
      <strong>{value}</strong>
      <div className="metric-foot"><span className="status-dot live" /> GraphShield engine</div>
    </div>
  );
}

function Panel({ title, subtitle, action, className = "", children }) {
  return (
    <section className={`panel ${className}`}>
      <div className="panel-head">
        <div>
          <h2>{title}</h2>
          <p>{subtitle}</p>
        </div>
        {action && <button className="panel-action">{action}</button>}
      </div>
      {children}
    </section>
  );
}

function App() {
  const [page, setPage] = useState("Overview");

  const Page = useMemo(() => {
    if (page === "Graph Explorer") return GraphExplorer;
    if (page === "Alerts") return Alerts;
    if (page === "Experiment Lab") return Experiments;
    return Overview;
  }, [page]);

  const nav = [
    ["Overview", "◈"],
    ["Graph Explorer", "◎"],
    ["Alerts", "⚠"],
    ["Experiment Lab", "◫"],
  ];

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">GS</div>
          <div>
            <strong>GraphShield</strong>
            <span>AI THREAT ENGINE</span>
          </div>
        </div>

        <nav>
          {nav.map(([label, icon]) => (
            <button
              key={label}
              className={page === label ? "nav-item active" : "nav-item"}
              onClick={() => setPage(label)}
            >
              <span>{icon}</span>
              {label}
            </button>
          ))}
        </nav>

        <div className="sidebar-bottom">
          <div className="engine-status">
            <div className="engine-top">
              <span>ENGINE STATUS</span>
              <span className="status-dot live" />
            </div>
            <strong>Operational</strong>
            <small>GATv2 · MPS inference</small>
          </div>
          <div className="sidebar-user">
            <div className="avatar">DS</div>
            <div>
              <strong>Research Console</strong>
              <span>Local workspace</span>
            </div>
          </div>
        </div>
      </aside>

      <main className="content">
        <header className="topbar">
          <div className="breadcrumbs">
            <span>GraphShield AI</span>
            <span>/</span>
            <strong>{page}</strong>
          </div>
          <div className="top-actions">
            <div className="demo-badge">DEMO DATA</div>
            <div className="top-icon">⌕</div>
            <div className="top-icon">⚙</div>
          </div>
        </header>

        <div className="page-content">
          <Page />
        </div>
      </main>
    </div>
  );
}

export default App;
